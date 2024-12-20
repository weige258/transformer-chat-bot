import time
import random
import threading

try:
    run_time=float(open("./time").read())
except:
    print("没有文件新建计时器记忆文件")
    run_time=0
    f=open("time",mode="w")
    f.write(str(run_time))

def time_func():
    while True:
        global run_time
        start_time=time.time()
        time.sleep(random.randint(100,500))
        end_time=time.time()
        run_time+=end_time-start_time
        f=open("time",mode="w")
        f.write(str(run_time))
        #输出时 分 秒
        hours = int(run_time // 3600)
        minutes = int((run_time % 3600) // 60)  
        seconds = int(run_time % 60)  
        print(f"运行时间 {hours}:{minutes}:{seconds}")

def run():  
    threading.Thread(target=time_func).start()




