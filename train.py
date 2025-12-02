import re
from main import *
import torch


f=open("train_sft.csv","r",encoding="utf-8")
text=f.read()
text=re.sub(pattern=r"\n",repl="",string=text)
pattern = r'<s>Human:(.*?)</s>'
a=re.findall(pattern,text,re.DOTALL)
pattern = r'<s>Assistant:(.*?)</s>'
b=re.findall(pattern,text,re.DOTALL)
print(f"数据量{len(b)}")
num=0


while True:
    try:
        i=random.randint(0,len(a))
        ask=a[i]
        answer=b[i]
        train(ask,answer)
        generation(ask)
        print("*"*100)
        num+=1
        if num%50==0:
            torch.save(obj=model,f="model.pth")
        else:
            continue
    except Exception as e:
        print(e)
        continue

