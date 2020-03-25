import numpy as np
import math 
import pickle as pk
from operator import itemgetter
from operator import attrgetter
import copy
import math
from functools import reduce
from math import gcd

import argparse
from threading import Thread
from subprocess import call

class task:
    def __init__(self, name ,priority, layer_num, period, release, cfg_list, default , glist, clist, deadline):
        self.name = name
        self.priority = priority
        self.layer_num = layer_num
        self.period = period
        self.release  = release
        self.cfg = cfg_list # resource allocation list
        self.default = default
        self.glist = glist # gpu execution time list
        self.clist = clist # cpu execution time list
        self.deadline = 0
        self.current_layer = 0 # currently executing layer number
        self.current_exec = 0 # currently executing layer execution time
        self.execed = 0  # currently exectued amount
        self.current_resource = 0 # currently allocated resource
        self.done = 1
        self.response = 0
        self.log = []
        self.logs = []
        self.g_util = sum(glist)/period
        self.c_util = sum(clist)/period
    def update_release(self):
        self.release += self.period
    def update_deadline(self):
        self.deadline += self.period
    def clear_up(self):
        self.release = 1
        self.deadline = 0
        self.current_layer = 0
        self.current_exec = 0
        self.execed = 0 
        self.current_resource = 0
        self.done =1
        self.cfg = copy.deepcopy(self.default)
        self.log = []
        self.logs = []

def LCM(a, b):
    return int(a * b / gcd(a, b))

def LCMS(taskset):
    periods = []
    for task in taskset:
        periods.append(task.period)
    periods = tuple(periods)
    return reduce(LCM, periods)

def darknet(task_info, result):
    task_name= task_info[0]
    task_period = int(task_info[1])
    task_num = int(task_info[2])
    
    if task_name == "Yolo":
        task_name = "./task/yolo.list"
    if task_name == "RNN":
        task_name = "./task/rnn.list"
    if task_name == "Resnet":
        task_name = "./task/resnet.list"
    if task_name == "Extraction":
        task_name = "./task/extraction.list"
    
    command_line = ["./darknet"]
    command_line.append("-task "+task_name)
    command_line.append("-period "+str(task_period))
    command_line.append("-task_num "+str(task_num))
    
    
    rev = call(command_line)
    result.append(rev)
    
def lalarand(task_num, mode):
    command_line = ["./LaLaRAND/lalarand"]
    command_line.append("-sync "+str(task_num))
    command_line.append("-mode "+str(mode))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type = int , default = 4, help = "1: ALL GPU 2: Preferable 3: Static 4: LaLaRAND")

    opt = parser.parse_args()

    fp = open("taskset.pk","rb")
    setoftaskset = pk.load(fp)
    fp.close()

    list_of_taskset_list = []
    for taskset in setoftaskset:
        lcm = LCMS(taskset)
        taskset_list = []
        for task in taskset:
            task_type = task.name[:-1]
            task_period = task.period/100
            task_num = lcm/task.period
            taskset_list.append([str(task_type), str(task_period), str(task_num)])
        list_of_taskset_list.append(taskset_list)
    
    sched = []
    unsched = []
    for taskset_list in list_of_taskset_list:
        task_num = len(taskset_list)
    
        task_thread = []
        result = []
        for task in taskset_list:
            task_thread.append(Thread(target = darknet, args= (task, result)))
        
        lalarand_thread = Thread(target = lalarand, args= (task_num ,opt.mode))
    
    
        lalarand_thread.start()
    
        for thread in task_thread:
            thread.start()
        
    
        for thread in task_thread:
            thread.join()
        
        lalarand_thread.join()
    

    if(sum(result) != task_num):
        unsched.append(taskset_list)
    else:
        sched.append(taskset_list)
    

    print("[sched] :", len(sched))
    print("[unsched] :", len(unsched))