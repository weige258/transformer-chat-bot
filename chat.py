from main import *

while True:
    try:
        user_input=input("输入聊天内容")
        generation(user_input)
    except:
        continue