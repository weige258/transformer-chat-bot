import random
total_loss=0
loss_num=0

def record_loss(loss:float):
    global total_loss,loss_num
    total_loss+=loss
    loss_num+=1
    if (random.randint(0,1000)==1):
        print("当前平均loss",total_loss/loss_num)
        total_loss = 0
        loss_num =0