import random
import torch
from  tool.loss_record import record_loss
from model import *

start_tensor=torch.tensor([1])
end_tensor=torch.tensor([10000])

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

def encode(text):
    tensor=torch.tensor([],dtype=torch.int)
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
    model=torch.load(f="model.pth",map_location=device,weights_only=False)
    print("载入模型")
except Exception as e:
    model=MainModel().to(device)
    print("新建模型",e)

loss_func=torch.nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)

def train(ask,answer):
    print("---训练问题 \n",ask)
    print("---训练回答 \n")
    
    ask_tensor=torch.cat((encode(ask),start_tensor)).to(device)
    answer_tensor=torch.cat((encode(answer),end_tensor)).to(device)
    tensor=torch.cat((ask_tensor,answer_tensor)).to(device)
    
    for i in range(len(ask_tensor),len(tensor)-1):
        input_tensor=tensor[0:i]
        autoregressive=input_tensor[-1].unsqueeze(0)
        label=probability(tensor[i]).to(device)
        
        print(chr(int(tensor[i])),end="")

        output=model(autoregressive,input_tensor)
        loss=loss_func(output,label)
        record_loss(float(loss))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
 
def generation(text):
    num=0
    output_text=""
    prompt = torch.cat((encode(text),start_tensor)).to(device)
    print("\n","---生成回复:","\n")
    for i in range(max_length):
        try:
            autoregressive=prompt[-1].to(device)
            output=model(autoregressive.unsqueeze(0),prompt)
            index = int(torch.multinomial(torch.softmax(output / temperature, dim=-1), 1))
            letter=chr(index).encode("utf-8").decode("utf-8")
            if index==int(start_tensor):
                letter=""
            if index==int(end_tensor):
                num+=1
                letter=""
            if num>=random.randint(1,3):
                break
            print(letter,end="")
            output_text+=letter
            prompt=torch.cat((prompt,torch.tensor([index]).to(device))).to(device)
        except:
            continue
    return output_text


