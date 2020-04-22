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
    int mode = find_int_arg(argc, argv, "-mode", 4); // mode 1: ALL GPU // mode 2: preferable // mode 3: Static //mode 4: LaLaRAND
    int index = find_int_arg(argc, argv, "-index", -1);

    printf("Sync : %d Mode :%d Index :%d\n", Sync, mode, index);

    if(index == -1){
        puts("taskset index is not correct!");
        exit(-1);
    }


    struct sched_param high;
    memset( &high, 0, sizeof(high));
    high.sched_priority = 20;
    if(sched_setscheduler(getpid(), SCHED_FIFO, &high) == -1) perror("SCHED_FIFO :");
    // cpu affininty setting 
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(1, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
    
    dnn_profile ** profile_list = make_profile_list(mode);

    dnn_queue * dnn_list = createDNNQueue();

    resource * gpu = createResource(GPU);
    resource * cpu = createResource(CPU);

    int reg_fd = open_channel(REGISTRATION, O_RDONLY | O_NONBLOCK);
    
    double current_time;
    int gpu_target, cpu_target;
    int fd_head;
    fd_set readfds;
    dnn_info *node;
    
    char log_path[50];
    switch (mode){
        case 1:
            snprintf(log_path, 50, "./Exp/RM/taskset_%d/lala_%d.txt", index, getpid());
            break;
        case 2:
            snprintf(log_path, 50, "./Exp/RM_PR/taskset_%d/lala_%d.txt", index, getpid());
            break;
        case 3:
            snprintf(log_path, 50, "./Exp/RM_DART/taskset_%d/lala_%d.txt", index, getpid());
            break;
        case 4:
            snprintf(log_path, 50, "./Exp/RM_LaLa/taskset_%d/lala_%d.txt", index, getpid());
    }
    
    freopen(log_path,"w", stderr);
    
    do{
        gpu_target = -1;
        cpu_target = -1;
        fd_head = make_fdset(&readfds, reg_fd, dnn_list);
        if(select(fd_head +1, &readfds, NULL, NULL, NULL)){
            current_time = get_time_point();
            // 1st registration check
            if(FD_ISSET(reg_fd, &readfds)) {
                check_registration(dnn_list, reg_fd);
                print_list("REGIST",dnn_list);
            }
            // 2nd request check 
            for(node = dnn_list ->head; node !=NULL; node = node -> next) 
                if(FD_ISSET(node->request_fd, &readfds))
                    request_handler(node, gpu, cpu, profile_list[node->type], current_time);

            print_queue("GPU",gpu->waiting);
            print_queue("CPU",cpu->waiting);
            if(!(gpu->waiting->count + cpu->waiting->count < Sync)){
                 
                if(Sync) update_deadline_all(dnn_list, current_time);
                
                double dequeue_start = get_time_point();
                if( gpu -> state == IDLE ) gpu_target = deQueue(gpu->waiting, dnn_list, profile_list, current_time, gpu);
                if( cpu -> state == IDLE ) cpu_target = deQueue(cpu->waiting, dnn_list, profile_list, current_time, cpu);
                fprintf(stderr, "[DEQUEUE] ABS : %f, Passed : %8.5f\n", dequeue_start,  ((double)get_time_point() - dequeue_start)/1000);

                if (mode == 4){ /* only in LaLaRAND */
                    if( gpu -> state == IDLE ) gpu_target = migration(cpu->waiting, dnn_list, profile_list, current_time, cpu, gpu);
                    if( cpu -> state == IDLE ) cpu_target = migration(gpu->waiting, dnn_list, profile_list, current_time, gpu, cpu);
                }
                
                double release_start = get_time_point();
                if(Sync) send_release_time(dnn_list);
                fprintf(stderr,"[RELEASE] ABS : %f\, Passed : %8.5f\n", release_start, ((double)get_time_point() - release_start)/1000);

                if(gpu_target != -1) decision_handler(gpu_target, dnn_list, GPU);
                if(cpu_target != -1) decision_handler(cpu_target, dnn_list, CPU);
                Sync = 0;
            }
            //printf("[OVERHEAD] LaLa %8.5f\n",((double)get_time_point() - current_time));
        }
    }while(!(Sync == 0 && dnn_list -> count == 0)); 
}   
