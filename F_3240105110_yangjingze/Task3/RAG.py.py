"""基于RAG的对话助手实现"""
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import json
import nltk
from nltk.tokenize import sent_tokenize

#nltk.download('punkt')

# ================ 全局配置 ================
DEVICE = "cpu"
EMBEDDING_MODEL = SentenceTransformer('all-mpnet-base-v2')  # 更强大的句子编码模型
LLM_MODEL = "flan-t5-base"  # 轻量级生成模型
MAX_HISTORY = 5  # 保留的对话历史轮数
KNOWLEDGE_ITEMS = 3  # 每次检索的知识片段数量


# ================ 工具类复用 ================
class Vectorizer:
    # 使用sentence-transformers模型
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

    def get_embeddings(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)


class KnowledgeBase:
    """知识库管理系统"""

    def __init__(self, vectorizer):
        """
        初始化知识库
        参数：
            vectorizer: 向量编码器实例
        """
        self.vectorizer = vectorizer
        self.texts = []  # 存储原始文本
        self.embeddings = np.empty((0, 768))  # 存储向量

    def add_documents(self, chunks):
        """
        添加文档到知识库
        参数：
            chunks: 预处理后的文本块列表
        """
        # 生成向量并添加到知识库
        new_embeddings = self.vectorizer.get_embeddings(chunks)
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.texts.extend(chunks)

    def save(self, base_path):
        """
        保存知识库到文件
        参数：
            base_path: 保存文件的基础路径（自动添加后缀）
        """
        np.save(f'{base_path}_embeddings.npy', self.embeddings)
        with open(f'{base_path}_texts.json', 'w') as f:
            json.dump(self.texts, f)

    @classmethod
    def load(cls, base_path, vectorizer):
        """
        从文件加载知识库
        参数：
            base_path: 文件基础路径
            vectorizer: 向量编码器实例
        """
        kb = cls(vectorizer)
        kb.embeddings = np.load(f'{base_path}_embeddings.npy')
        with open(f'{base_path}_texts.json', 'r') as f:
            kb.texts = json.load(f)
        return kb


class Retriever:
    """语义检索器"""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def query(self, question):
        """
        执行语义检索查询
        参数：
            question: 查询语句
            top_k: 返回最相似的前k个结果
        返回：
            包含(文本, 相似度得分)的元组列表
        """
        # 编码查询语句
        query_embedding = self.kb.vectorizer.get_embeddings([question])

        # 计算余弦相似度
        similarities = cosine_similarity(query_embedding, self.kb.embeddings)

        # 获取相似度最高的索引
        top_indice = similarities.argsort()[0][-1:][::-1]

        # 返回结果列表
        return (self.kb.texts[top_indice[0]], similarities[0][top_indice[0]])


# ================ RAG对话核心类 ================
class RAGAssistant:
    def __init__(self, knowledge_base):
        """
        初始化对话助手
        参数：
            knowledge_base: 已加载的知识库实例
        """
        self.kb = knowledge_base
        self.retriever = Retriever(knowledge_base)
        self.history = []

        # 初始化生成模型
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL).to(DEVICE)
        self.generator = pipeline(
            "text2text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
            device=-1
        )

    def _build_prompt(self, question):
        """构建包含上下文和历史的提示"""
        # 检索相关知识
        knowledge = self.retriever.query(question, top_k=KNOWLEDGE_ITEMS)
        knowledge_text = "".join([f"[知识]{text}" for text, _ in knowledge])

        # 构建历史
        history_text = ""
        if self.history:
            history_text = "".join([f"用户：{q}助手：{a}" for q, a in self.history[-MAX_HISTORY:]])

            return f"""基于以下知识和对话历史回答最后的问题：{knowledge_text}{history_text}用户：{question}助手："""


def ask(self, question):
    """处理用户提问"""
    # 生成提示
    prompt = self._build_prompt(question)

    # 生成回答
    response = self.generator(
        prompt,
        max_length=200,
        num_beams=5,
        early_stopping=True,
        temperature=0.7
    )[0]['generated_text']

    # 更新历史记录
    self.history.append((question, response))

    # 保持历史长度
    if len(self.history) > MAX_HISTORY:
        self.history = self.history[-MAX_HISTORY:]

    return response


# ================ 完整使用流程 ================
if __name__ == "__main__":
    # 初始化知识库
    vectorizer = Vectorizer()
    try:
        # 尝试加载已有知识库
        kb = KnowledgeBase.load("my_knowledge_base", vectorizer)
        print("成功加载已有知识库")
    except:
        # 新建知识库
        print("新建知识库...")
        chunks = preprocess_text("sample_document.txt")
        kb = KnowledgeBase(vectorizer)
        kb.add_documents(chunks)
        kb.save("my_knowledge_base")

    # 初始化助手
    assistant = RAGAssistant(kb)

    print("欢迎使用知识库助手（输入'exit'退出）")
    while True:
        user_input = input("用户：")
        if user_input.lower() == 'exit':
            break

        # 获取并显示回答
        response = assistant.ask(user_input)
        print(f"助手：{response}")

print("对话已结束")
