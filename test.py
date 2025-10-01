import re
import random
import main

f=open("train_sft.csv","r",encoding="utf-8")
text=f.read()
text=re.sub(pattern=r"\n",repl="",string=text)
pattern = r'<s>Human:(.*?)</s>'
a=re.findall(pattern,text,re.DOTALL)
pattern = r'<s>Assistant:(.*?)</s>'
b=re.findall(pattern,text,re.DOTALL)

while True:
    try:
        i=random.randint(0,len(a))
        ask=a[i]
        answer=b[i]
        print(ask)
        main.generation(ask)
        print("*"*100)
    except Exception as e:
        print(e)
        continue