"""
基于BERT的知识库系统完整实现
包含：文本预处理、向量编码、知识库构建、语义检索等功能
"""

# ==================== 导入依赖 ====================
import numpy as np
import json
import torch
import nltk
import re
from nltk.tokenize import sent_tokenize
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# ==================== 初始化NLTK ====================
#nltk.download('punkt')  # 下载句子分割所需的punkt数据
#nltk.download('punkt_tab')

# ==================== 全局配置 ====================
DEVICE = torch.device('cpu') # 使用cpu
BERT_MODEL_NAME = 'bert-base-uncased'  # 使用的BERT模型名称
CHUNK_SIZE = 150  # 文本块的最大单词数


# ==================== 数据预处理模块 ====================
def preprocess_text(file_path, chunk_size=CHUNK_SIZE):
    """
    文本预处理函数：将长文本分割为语义连贯的文本块
    参数：
        file_path: 文本文件路径
        chunk_size: 每个文本块的最大单词数
    返回：
        chunks: 分割后的文本块列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    #sentences = sent_tokenize(text)     # 使用NLTK进行句子分割
    sentences = re.split('(。|！|\！|\.|？|\?|\n)', text)      # 根据符号对句子进行分割
    #print(len(sentences))

    chunks = []  # 最终生成的文本块列表
    current_chunk = []  # 当前正在构建的文本块
    current_length = 0  # 当前文本块的单词计数

    for sent in sentences:
        sent_length = len(sent.split())  # 计算当前句子的单词数

        # 如果当前块还能容纳这个句子
        if current_length + sent_length <= chunk_size:
            current_chunk.append(sent)
            current_length += sent_length
        else:
            # 将当前块合并为字符串并保存
            chunks.append(' '.join(current_chunk))
            current_chunk = [sent]  # 开始新的文本块
            current_length = sent_length

    # 添加最后一个未完成的文本块
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    #print(len(chunks))

    return chunks


# ==================== 向量编码模块 ====================
class Vectorizer:
    # BERT文本向量化处理器

    def __init__(self):
        # 初始化BERT的分词器和模型
        self.tokenizer = BertTokenizer.from_pretrained(r"D:\F\bert-base-chinese")
        self.model = BertModel.from_pretrained(r"D:\F\bert-base-chinese").to(DEVICE)
        self.model.eval()  # 设置为评估模式

    def get_embeddings(self, texts):
        """
        将文本列表编码为BERT向量
        参数：
            texts: 需要编码的文本列表
        返回：
            numpy数组形状为(len(texts), 768)的向量矩阵
        """
        # 文本分词和编码
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,  # BERT的最大输入长度
            return_tensors="pt"
        ).to(DEVICE)

        # 前向传播（禁用梯度计算以节省内存）
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 使用平均池化获取文本向量
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings


# ==================== 知识库核心类 ====================
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


# ==================== 检索器类 ====================
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


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 初始化向量编码器
    vectorizer = Vectorizer()

    # 示例文档路径（替换为实际路径）
    doc_path = "data.txt"

    # 步骤1：预处理文档
    chunks = preprocess_text(doc_path)

    # 步骤2：构建知识库
    kb = KnowledgeBase(vectorizer)
    kb.add_documents(chunks)
    kb.save("my_knowledge_base")

    # 步骤3：加载知识库
    loaded_kb = KnowledgeBase.load("my_knowledge_base", vectorizer)

    # 初始化检索器
    retriever = Retriever(loaded_kb)

    # 执行查询
    question = input("\n请输入你的问题：\n")  #输入想要问的问题
    results = retriever.query(question)

    # 打印结果
    print(f"查询：'{question}'")
    print(results[0][:80] + "...")  # 显示前80个字符