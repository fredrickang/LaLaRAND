import math
from functools import reduce
from math import gcd
import argparse
from threading import Thread
import subprocess 
import os, signal

def darknet(task_info, pids, result):
    task_name= task_info[0]
    task_period = int(task_info[1])
    task_num = int(task_info[2])
    
    if task_name == "Yolo":
        task_name = "./task/yolotiny.list"
    if task_name == "RNN":
        task_name = "./task/rnn.list"
    if task_name == "Resnet":
        task_name = "./task/resnet18.list"
    if task_name == "Extraction":
        task_name = "./task/extraction.list"
    command_line = ["./darknet"]
    command_line.append("-task")
    command_line.append(task_name)
    command_line.append("-period")
    command_line.append(str(task_period))
    command_line.append("-num")
    command_line.append(str(task_num))
    
    sub = subprocess.Popen(command_line)
    
    pid = sub.pid
    pids.append(pid)
    
    sub.wait()
    if sub.returncode != 0:
        for pid in pids:
            try:
                os.kill(pid, 0)
            except OSError:
                pass
            else:
                os.kill(pid,signal.SIGKILL)
        result.append(-1)
    else:
        result.append(1) 
    
def lalarand(task_num, lalarand_pid ,mode):
    command_line = ["./lalarand/lalarand"]
    command_line.append("-sync")
    command_line.append(str(task_num))
    command_line.append("-mode")
    command_line.append(str(mode))
   
    sub = subprocess.Popen(command_line)
    lalarand_pid.append(sub.pid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type = int , default = 4, help = "1: ALL GPU 2: Preferable 3: Static 4: LaLaRAND")

    opt = parser.parse_args()

    fp = open("taskset_list.txt","r")
    lines = fp.readlines()
    fp.close() 

    list_of_taskset_list = []
    taskset_list = []
    for line in lines[1:]:
        token = line.split()
        if len(token) == 1:
            list_of_taskset_list.append(taskset_list)
            taskset_list = []
        else:
            taskset_list.append(token)
    
    sched = []
    unsched = []
    for taskset_list in list_of_taskset_list:
        task_num = len(taskset_list)
        task_thread = []
        
        result = []
        pids = []

        lalarand_pid = []
        lalarand_thread = Thread(target = lalarand, args= (task_num, lalarand_pid ,opt.mode))
       
        for task in taskset_list:
            task_thread.append(Thread(target = darknet, args= (task, pids ,result)))
    
    
        lalarand_thread.start()
    
        for thread in task_thread:
            thread.start()
        
    
        for thread in task_thread:
            thread.join()
        if(sum(result) != task_num):
            unsched.append(taskset_list)
        else:
            sched.append(taskset_list)

        os.kill(lalarand_pid[0],signal.SIGKILL)

        lalarand_thread.join()


    print("[sched] :", len(sched))
    print("[unsched] :", len(unsched))
