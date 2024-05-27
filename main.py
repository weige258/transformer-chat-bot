import random

import torch

from model import *

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

def encode(text):
    tensor=torch.tensor([],dtype=torch.long)
    for letter in text:
        try:
            tensor=torch.cat((tensor,torch.tensor([ord(letter)])))
        except:
            continue
    return tensor

def probability(letter_tensor):
    tensor=torch.zeros(dict_size)
    try:
        tensor[letter_tensor]=1
    except:
        pass
    return tensor

try:
    model=torch.load(f="model.pth",map_location=device)
    print("载入模型")
except:
    model=MainModel().to(device)
    print("新建模型")
loss_func=torch.nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.SGD(model.parameters(),lr=1e-4)

def train(ask,answer):
    prompt=encode(ask).to(device)
    answer_tensor= torch.cat((encode(answer),torch.tensor([10000]))).to(device)
    for next in answer_tensor:
        autoregressive=prompt[-1].to(device)
        label=probability(next).to(device)
        output=model(autoregressive.unsqueeze(0),prompt)
        loss=loss_func(output,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prompt=torch.cat((prompt,next.unsqueeze(0))).to(device)

def generation(text):
    num=0
    output_text=""
    prompt = encode(text).to(device)
    for i in range(max_length):
        try:
            autoregressive=prompt[-1].to(device)
            output=model(autoregressive.unsqueeze(0),prompt)
            index = int(torch.multinomial(torch.softmax(output / temperature, dim=-1), 1))
            letter=chr(index).encode("utf-8").decode("utf-8")
            if index==10000:
                num+=1
                letter=""
            if num>random.randint(1,4):
                break
            output_text+=letter
            prompt=torch.cat((prompt,torch.tensor([index]).to(device))).to(device)
        except:
            continue
    print(output_text)
    return output_text

