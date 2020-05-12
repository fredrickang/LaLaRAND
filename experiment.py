import math
from functools import reduce
from math import gcd
import argparse
from threading import Thread
import subprocess 
import os, signal

def darknet(task_info, pids, lalarand_pid ,result, mode, index, log_path):
    task_name= task_info[0]
    task_priority = int(task_info[1])
    task_period = int(task_info[2])
    task_num = int(task_info[3])
    
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
    command_line.append("-priority")
    command_line.append(str(task_priority))
    command_line.append("-period")
    command_line.append(str(task_period))
    command_line.append("-num")
    command_line.append(str(task_num))
    command_line.append("-mode")
    command_line.append(str(mode))
    command_line.append("-index")
    command_line.append(str(index))

    sub = subprocess.Popen(command_line)
    
    pid = sub.pid
    pids.append(pid)
    
    sub.wait()
    if sub.returncode != 0:
        for pid in lalarand_pid:
            try:
                os.kill(pid, 0)
            except OSError:
                pass
            else:
                os.kill(pid,signal.SIGKILL)

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




def lalarand(task_num, lalarand_pid ,mode, index):
    command_line = ["./lalarand/lalarand"]
    command_line.append("-sync")
    command_line.append(str(task_num))
    command_line.append("-mode")
    command_line.append(str(mode))
    command_line.append("-index")
    command_line.append(str(index))
    sub = subprocess.Popen(command_line)
    lalarand_pid.append(sub.pid)

def submain(mode, _list, path, start, end):

    fp = open(_list,"r")
    lines = fp.readlines()
    fp.close() 

    f_sched = open(path+"Sched.txt","w")
    f_unsched = open(path+"Unsched.txt","w")

    list_of_taskset_list = []
    taskset_list = []
    
    for line in lines:
        token = line.split()
        if len(token) == 1:
            list_of_taskset_list.append(taskset_list)
            taskset_list = []
        else:
            taskset_list.append(token)
    
    sched = []
    unsched = []
    
    if end == -1:
        end = len(list_of_taskset_list)
    
    print(end - start)
    for i, taskset_list in enumerate(list_of_taskset_list[start:end]):
        index = start + i
        taskset_path = os.path.join(path,"taskset_"+str(index))
        
        try:
            os.mkdir(taskset_path)
        except FileExistsError:
            pass

        fp = open(taskset_path + "/tasksetinfo.txt","w")

        for task in taskset_list:
            for info in task:
                fp.write(str(info)+" ")
            fp.write("\n")


        task_num = len(taskset_list)
        task_thread = []
        
        result = []
        pids = []

        lalarand_pid = []
        lalarand_thread = Thread(target = lalarand, args= (task_num, lalarand_pid ,mode, index))
       
        for task in taskset_list:
            task_thread.append(Thread(target = darknet, args= (task, pids, lalarand_pid ,result, mode, index, path)))
    
    
        lalarand_thread.start()
    
        for thread in task_thread:
            thread.start()
        
    
        for thread in task_thread:
            thread.join()
    
        if(sum(result) != task_num):
            fp.write("unsched\n")
            f_unsched.write(str(index) + "\n")
            unsched.append(taskset_list)
            for task in taskset_list:
                for element in task:
                    f_unsched.write(element)
                    f_unsched.write(" ")
                f_unsched.write("\n")
        else:
            fp.write("sched\n")
            f_sched.write(str(index) + "\n")
            sched.append(taskset_list)
            for task in taskset_list:
                for element in task:
                    f_sched.write(element)
                    f_sched.write(" ")
                f_sched.write("\n")


        fp.close()

        os.kill(lalarand_pid[0],signal.SIGKILL)

        lalarand_thread.join()


    print("[sched] :", len(sched))
    print("[unsched] :", len(unsched))

import shutil

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    #parser.add_argument("--mode", type = int , default = 4, help = "1: ALL GPU 2: Preferable 3: Static 4: LaLaRAND")
    #parser.add_argument("--start",type = int , default = 0 )
    #parser.add_argument("--n", type = int, default = -1, help = " -1 : ALL , other is other number")
    #parser.add_argument("--log_path", type = str, default = "Exp/RM/")
    #parser.add_argument("--list", type = str, default = "taskset_list.txt")
    opt = parser.parse_args()
    
    print(opt)
    task_num_list = [10, 11, 12]
    util_list = [0.8, 0.9, 1.0, 1.1, 1.2]

    for task_num in task_num_list:
        for util in util_list:
            if task_num == 10 and util <= 1.0:
                continue
            path = "taskset_"+str(task_num)
            path_detail = path + "_" + str(util) + ".txt"

            full_path = os.path.join(path, path_detail)

            submain(1,full_path,"./Exp/RM/",0, -1)
            submain(4,full_path,"./Exp/RM_LaLa/", 0, -1)

            shutil.move("./Exp", "../TEST_LOG/Single_Core_TNUM_UTIL/"+path+"_"+str(util))
            os.mkdir("./Exp")
            os.mkdir("./Exp/RM")
            os.mkdir("./Exp/RM_LaLa")
