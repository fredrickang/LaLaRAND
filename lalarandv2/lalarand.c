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

#define GPU 1

#define CPU 0

#define REGISTRATION "/tmp/lalarand_registration"

int main(int argc, char **argv){
    int Sync = find_int_arg(argc, argv, "-sync", 1);
    int baseline = find_int_arg(argc, argv, "-baseline", 1); // mode 1: ALL GPU // mode 2: preferable // mode 3: DART 
    int algo = find_int_arg(argc, argv, "-algo", 0);
    int index = find_int_arg(argc, argv, "-index", -1);
    
    printf("Sync : %d Baseline :%d Algo :%d Index :%d\n", Sync, baseline, algo, index);

    if(index == -1){
        puts("taskset index is not correct!");
        exit(-1);
    }


    struct sched_param high;
    memset( &high, 0, sizeof(high));
    high.sched_priority = 80;
    
    if(sched_setscheduler(getpid(), SCHED_FIFO, &high) == -1) perror("SCHED_FIFO :");
    // cpu affininty setting 
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
    
    dnn_profile ** profile_list = make_profile_list();

    dnn_queue * dnn_list = createDNNQueue();

    resource * gpu = createResource(GPU);
    resource * cpu = createResource(CPU);

    int reg_fd = open_channel(REGISTRATION, O_RDONLY | O_NONBLOCK);
    
    double current_time;
    int gpu_target, cpu_target;
    int fd_head;
    fd_set readfds;
    dnn_info *node;
    
    char log_path[60];
    if(DEBUG){
        switch (baseline){
            case 1:
                if(algo) snprintf(log_path, 60, "./Exp/ALL_LaLa/taskset_%d/lala_%d.txt", index, getpid());
                else snprintf(log_path, 60, "./Exp/ALL/taskset_%d/lala_%d.txt", index, getpid());
                break;
            case 2:
                if(algo) snprintf(log_path, 50, "./Exp/PR_LaLa/taskset_%d/lala_%d.txt", index, getpid());
                else snprintf(log_path, 50, "./Exp/PR/taskset_%d/lala_%d.txt", index, getpid());
                break;
            case 3:
                if(algo) snprintf(log_path, 50, "./Exp/DART_ALL_LaLa/taskset_%d/lala_%d.txt", index, getpid());
                else snprintf(log_path, 50, "./Exp/DART_ALL/taskset_%d/lala_%d.txt", index, getpid());
                break;
            case 4:
                if(algo) snprintf(log_path, 50, "./Exp/DART_GC_LaLa/taskset_%d/lala_%d.txt", index, getpid());
                else snprintf(log_path, 50, "./Exp/DART_GC/taskset_%d/lala_%d.txt", index, getpid());
                break;
            case 5:
                if(algo) snprintf(log_path, 50, "./Exp/DART_CG_LaLa/taskset_%d/lala_%d.txt", index, getpid());
                else snprintf(log_path, 50, "./Exp/DART_CG/taskset_%d/lala_%d.txt", index, getpid());
        }
    
        freopen(log_path,"w", stderr);
    }

    
    do{
        gpu_target = -1;
        cpu_target = -1;
        fd_head = make_fdset(&readfds, reg_fd, dnn_list);
        if(select(fd_head +1, &readfds, NULL, NULL, NULL)){
            current_time = get_time_point();
            // 1st registration check
            if(FD_ISSET(reg_fd, &readfds)) {
                check_registration(dnn_list, reg_fd, baseline, profile_list);
                print_list("REGIST",dnn_list);
            }
            // 2nd request check 
            for(node = dnn_list ->head; node !=NULL; node = node -> next) 
                if(FD_ISSET(node->request_fd, &readfds))
                    request_handler(node, gpu, cpu, current_time);

            print_queue("GPU",gpu->waiting);
            print_queue("CPU",cpu->waiting);
            
            if(!(gpu->waiting->count + cpu->waiting->count < Sync)){
                if(Sync) update_deadline_all(dnn_list, current_time);
                
                if(algo){
                    if ( (cpu -> state == IDLE) && (cpu->waiting->count == 0) && (gpu->waiting->count != 0) ){
                        cpu_target = migration(gpu->waiting, dnn_list, profile_list, current_time, gpu, cpu);
                        if (cpu -> state == IDLE){
                            cpu_target = sacrifice(gpu->waiting, dnn_list, profile_list, current_time, gpu, cpu);
                        }
                    }
                    if ( (gpu -> state == IDLE) && (gpu->waiting->count == 0) && (cpu->waiting->count != 0) ){
                        gpu_target = migration(cpu->waiting, dnn_list, profile_list, current_time, cpu, gpu);
                        if (gpu-> state == IDLE) {
                            gpu_target = sacrifice(cpu->waiting, dnn_list, profile_list, current_time, cpu, gpu);
                        }
                    }
                    if( gpu -> state == IDLE) gpu_target = deQueue_algo(gpu->waiting, dnn_list, profile_list, current_time, gpu);
                    if( cpu -> state == IDLE) cpu_target = deQueue_algo(cpu->waiting, dnn_list, profile_list, current_time, cpu);
                }
                
                if( gpu -> state == IDLE ) gpu_target = deQueue(gpu->waiting, dnn_list, profile_list, current_time, gpu);
                if( cpu -> state == IDLE ) cpu_target = deQueue(cpu->waiting, dnn_list, profile_list, current_time, cpu);
                
                if(Sync) send_release_time(dnn_list);

                if(gpu_target != -1) decision_handler(gpu_target, dnn_list, GPU);
                if(cpu_target != -1) decision_handler(cpu_target, dnn_list, CPU);
                Sync = 0;
            }
        }
    }while(!(Sync == 0 && dnn_list -> count == 0)); 
}   
