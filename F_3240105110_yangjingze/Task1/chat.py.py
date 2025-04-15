import requests  #导入request包

# 设置API密钥
api_key = 'sk-UHc6pcAcbTWZFTReSdiALfGQvw8r2i0ivD5Y5AYcpS4Z2tkr'

# 创建API请求URL
url = 'https://api.zchat.tech/v1/chat/completions'

# 定义一系列headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json", #json格式
}

chat_history = []  #定义一个列表，用来存和GPT的对话历史

#定义一个访问API的函数
def use_api(chat_history):
    # 定义传给API的data
    data = {
        "model": "gpt-3.5-turbo",  # 选择模型
        "messages": chat_history  # 传入历史问的问题及回答
    }

    response = requests.post(url, headers=headers, json=data)  #把问题发送给API

    return response

#定义一个打印和保存答案的函数n
def save_answer(response):
    if response.status_code == 200:  # 访问成功
        result = response.json()
        answer = result["choices"][0]["message"]["content"]  #API最近一次返回的答案
        print(answer) #打印最近的一次答案
        chat_history.append({"role": "assistant", "content": answer})  # 把AI的答案加到列表里
    else:
        print(f"Error: {response.status_code}")  #打印错误信息


    '''#把答案保存到一个txt文件
    with open('answer.txt', 'w', encoding="utf-8") as f:
        f.write(answer)
    f.close()'''


def main():
    while True: #不断循环执行
        question = input("\n请输入你的问题（输入”拜拜“结束对话）：\n")  #输入想要问的问题
        if question == "拜拜":
            print("再见！")
            break  #退出while循环
        chat_history.append({"role": "user", "content": question})  # 把我的提问加到列表里
        response = use_api(chat_history)
        save_answer(response)
        if response.status_code != 200:  # 访问失败
            break  #退出while循环



if __name__ == "__main__":
    main()

