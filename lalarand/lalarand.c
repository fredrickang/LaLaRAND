#define DEBUG 0
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <fcntl.h>
#include <sys/types.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <signal.h>
#include <time.h>
#include <chrono>
#include <float.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "lalarand.h"
#include "lalarand_fn.h"

#define MEM 2
#define GPU 1
#define CPU 0

#define REGISTRATION "/tmp/lalarand_registration"

int main(int argc, char **argv){
    int Sync = find_int_arg(argc, argv, "-sync", 1);
    int baseline = find_int_arg(argc, argv, "-baseline", 1); // mode 1: ALL GPU // mode 2: preferable // mode 3: DART 
    int algo = find_int_arg(argc, argv, "-algo", 0);
    int index = find_int_arg(argc, argv, "-index", -1);
    int hiding = find_int_arg(argc,argv, "-hiding", 0);

    printf("Sync : %d Baseline :%d Algo :%d Index :%d\n", Sync, baseline, algo, index);

    if(index == -1){
        puts("taskset index is not correct!");
        exit(-1);
    }

    set_priority(50); 
    set_affinity(0);
    
    dnn_profile ** profile_list = make_profile_list(baseline);
    dnn_queue * dnn_list = createDNNQueue();
    resource * gpu = createResource(GPU);
    resource * cpu = createResource(CPU);
    resource * mem = createResource(MEM);

    int reg_fd = open_channel(REGISTRATION, O_RDONLY | O_NONBLOCK);
    
    double current_time;
    int gpu_target, cpu_target, mem_target;
    int fd_head;
    fd_set readfds;
    dnn_info *node;
    
    if(DEBUG)logging(baseline, algo, index);
    
    do{
        gpu_target = -1;
        cpu_target = -1;
        mem_target = -1;

        fd_head = make_fdset(&readfds, reg_fd, dnn_list);
    
        if(select(fd_head +1, &readfds, NULL, NULL, NULL)){
            current_time = get_time_point();
            if(FD_ISSET(reg_fd, &readfds)) {
                check_registration(dnn_list, reg_fd, gpu, cpu,baseline);
                print_list("REGIST",dnn_list);
            }
            // 2nd request check 
            for(node = dnn_list ->head; node !=NULL; node = node -> next) 
                if(FD_ISSET(node->request_fd, &readfds))
                    request_handler(hiding, node, gpu, cpu, mem, profile_list[node->type], current_time);

            print_queue("GPU",gpu->waiting);
            print_queue("CPU",cpu->waiting);
            print_queue("MEM",mem->waiting);
            if(!(gpu->waiting->count + cpu->waiting->count < Sync)){
                if(Sync) update_deadline_all(dnn_list, current_time);

                if(hiding){
                    if(mem -> state == IDLE) mem_target = deQueue_mem(mem->waiting, current_time, mem, gpu, cpu);
                    if(gpu -> state == IDLE) gpu_target = deQueue_hiding(gpu->waiting, current_time, gpu, mem);
                    if(cpu -> state == IDLE) cpu_target = deQueue_hiding(cpu->waiting, current_time, cpu, mem);
                }
                else{
                    if(gpu -> state == IDLE) gpu_target = deQueue(gpu->waiting, current_time, gpu);
                    if(cpu -> state == IDLE) cpu_target = deQueue(cpu->waiting, current_time, cpu);
                }

                if(algo){
                    if(gpu -> state == IDLE) gpu_target = migration(cpu->waiting, dnn_list, profile_list, current_time, cpu, gpu);
                    if(cpu -> state == IDLE) cpu_target = migration(gpu->waiting, dnn_list, profile_list, current_time, gpu, cpu);
                }

                if(Sync) send_release_time(dnn_list);

                if(gpu_target != -1) decision_handler(gpu_target, dnn_list, GPU);
                if(cpu_target != -1) decision_handler(cpu_target, dnn_list, CPU);
                if(mem_target != -1) decision_handler(mem_target, dnn_list, MEM);
                Sync = 0;
            }
        }
    }while(!(Sync == 0 && dnn_list -> count == 0)); 
}   
