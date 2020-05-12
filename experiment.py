import math
from functools import reduce
from math import gcd
import argparse
from threading import Thread
import subprocess 
import os, signal

def darknet(task_info, pids, lalarand_pid ,result, mode, index, log_path, cut):
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
    if (mode == 6 or mode ==5):
        command_line.append("-cut")
        command_line.append(str(cut))

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

        if mode != 5 and mode != 6:
            for task in taskset_list:
                task_thread.append(Thread(target = darknet, args= (task, pids, lalarand_pid ,result, mode, index, path, -2)))
        else:
            cut_list = generate_dart_cut(taskset_list)
            for i, task in enumerate(taskset_list):
                task_thread.append(Thread(target = darknet, args = (task, pids, lalarand_pid, result, mode, index, path , cut_list[i])))
    
        lalarand_thread.start()
    
        for thread in task_thread:
            thread.start()
        
    
        for thread in task_thread:
            thread.join()
    
        if(sum(result) != task_num):
            fp.write("unsched\n")
            unsched.append(taskset_list)
            for task in taskset_list:
                for element in task:
                    f_unsched.write(element)
                    f_unsched.write(" ")
                f_unsched.write("\n")
            f_unsched.write(str(index) + "\n")
        else:
            fp.write("sched\n")
            sched.append(taskset_list)
            for task in taskset_list:
                for element in task:
                    f_sched.write(element)
                    f_sched.write(" ")
                f_sched.write("\n")
            f_sched.write(str(index) + "\n")

        fp.close()

        os.kill(lalarand_pid[0],signal.SIGKILL)

        lalarand_thread.join()


    print("[sched] :", len(sched))
    print("[unsched] :", len(unsched))

from operator import itemgetter
import sys
def avg_util(gpu, cpu, period):
    return (sum(gpu) + sum(cpu))/(period * 2)

def M(task, mode, weight):
    gpu = task[0]
    cpu = task[1]
    period  = task[3]
        
    smallest_diff = sys.maxsize
    smallest_fist = 0
    smallest_secod = 0
    smallest_index = 0
    for index in range(len(gpu) + 1):
        if mode == 5:
            first = weight[0] + gpu[:index]/period
            second = weight[1] + cpu[index:]/period
        else: 
            first = weight[0] + cpu[:index]/period
            second = weight[1] + gpu[index:]/period
        diff = abs(first - second)
        if diff < smallest_diff:
            smallest_diff = diff
            smallest_index = index
            smallest_fist = first
            smallest_secod = second
    return smallest_index, [smallest_fist, smallest_secod]

def generate_dart_cut(taskset_list, mode):
    rev = []
    yolo_g = [305,  37, 124,  21,  74,  17,  59,  14,  52,   9,  60,  12, 182, 39,  55,  27,  44,   5,  31,  14,  12, 101,  31,  48]
    yolo_c = [3287,   698,  6423,   352,  9595,   323,  6334,   175,  6590, 54,  8118,   104, 32054,  1894,  8062,   967,   101,    11, 366,    70,    54, 19150,  1530,   388]
    
    extrac_g = [94,  28,  84,  15,  37,  53,  40, 144,  13,  30,  57,  30,  56, 30,  56,  31,  56,  37, 184,  11,  28, 132,  29, 132,  35,   9, 50,   2]
    extrac_c = [7068,   210, 10416,   162,   727,  7156,  1645, 28586,   113, 1020,  7878,  1011,  7860,  1009,  7853,  1009,  7860,  1864, 30857, 62, 1816, 15826, 1797, 15833, 3493, 14, 6, 1]
    
    resnet_g = [113,  30,  47,  40,  18,  43,  47,  16,  41,  57,  15,  57,  54, 11,  36,  36,  13,  35,  34,   9,  44,  70,  16,  72,  75,  10, 13,  26,  49]
    resnet_c = [7398,  271, 4892, 4752,   65, 4770, 4726,   65, 2460, 4914,   86, 4954, 4949,   35, 2609, 5131,   40, 5116, 5102,   22, 3702, 7235, 25, 7283, 7302,   12,   11,  374,    6]
    
    rnn_g = [130, 113, 107,  28,  20,   3]
    rnn_c = [113, 139, 139,  16,   5,   4]

    new_list = []
    for i, task in enumerate(taskset_list):
        tmp = []
        if task[0] == "Yolo":
            tmp.append(yolo_g)
            tmp.append(yolo_c)
            tmp.append(avg_util(yolo_g, yolo_c, int(task[2])))

        if task[0]== "RNN":
            tmp.append(rnn_g)
            tmp.append(rnn_c)
            tmp.append(avg_util(rnn_g, rnn_c, int(task[2])))
        
        if task[0] == "Resnet":
            tmp.append(resnet_g)
            tmp.append(resnet_c)
            tmp.append(avg_util(resnet_g,resnet_c, int(task[2])))
        
        if task[0] == "Extraction":
            tmp.append(extrac_g)
            tmp.append(extrac_c)
            tmp.append(avg_util(extrac_g, extrac_c, int(task[2])))
        
        tmp.append(int(task[2]))
        tmp.append(i)
        new_list.append(tmp)

    new_list = sorted(new_list, key=itemgetter(2))
    
    weight = [0,0]
    for task in new_list:
        index ,weigth = M(task,mode, weight)
        task.append(index)
    
    new_list = sorted(new_list, key=itemgetter(-2))
    for task in new_list:
        rev.append(task[-1])
    return rev

import shutil

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type = int , default = 4, help = "1: ALL GPU 2: Preferable 3: LaLaRAND 4: DART")
    parser.add_argument("--list", type = str, default = "taskset_list.txt")
    parser.add_argument("--start",type = int , default = 0 )
    parser.add_argument("--end", type = int, default = -1, help = " -1 : ALL , other is other number")
    parser.add_argument("--log_path", type = str, default = "Exp/RM/")

    opt = parser.parse_args()
    
    print(opt)

    if(opt.mode != 4):
        submain(opt.mode, opt.list, opt.log_path, opt.start, opt.end)
    else: # dart 1. RM 2. GC 3. CG
        submain(1, opt.list, "./Exp/RM/", opt.start, opt.end)
        submain(5, "./Exp/RM/Unsched.txt", "./Exp/RM_GC/", 0, -1)
        submain(6, "./Exp/RM_GC/Unsched.txt", "./Exp/RM_CG/", 0, -1)
