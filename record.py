import threading
import time

def hours_minutes_seconds_to_seconds(seconds:str)->int:
    h,m,s=seconds.split(":")
    return int(h)*3600+int(m)*60+int(s)

def seconds_to_hours_minutes_seconds(seconds:int)->str:
    h=seconds//3600
    m=(seconds%3600)//60
    s=seconds%60
    return f"{h:02}:{m:02}:{s:02}"

runnning_time=0
total_loss=0
record_count=0

def load_run_time():
    global runnning_time
    try:
        with open("record.txt","r",encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                # 解析运行时间
                time_start = last_line.find("<time>") + len("<time>")
                time_end = last_line.find("</time>")
                time_str = last_line[time_start:time_end]
                runnning_time = hours_minutes_seconds_to_seconds(time_str)

    except:
        with open("record.txt","w",encoding="utf-8") as f:
            f.write("<time>00:00:00</time><avg_loss>0</avg_loss>\n")

load_run_time()

def time_thread():
    global runnning_time
    while True:
        runnning_time+=1
        time.sleep(1)

threading.Thread(target=time_thread,daemon=True).start()

def record_loss(loss:float):
    try:
        global runnning_time,total_loss,record_count
        total_loss+=loss
        record_count+=1
        if record_count%10000==9999:
            avg_loss=total_loss/record_count
            with open("record.txt","a",encoding="utf-8") as f:
                f.write(f"<time>{seconds_to_hours_minutes_seconds(runnning_time)}</time><avg_loss>{avg_loss}</avg_loss>\n")
            total_loss=0
            record_count=0
    except Exception as e:
        print(e)



    