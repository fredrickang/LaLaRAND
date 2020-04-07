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

    printf("Sync : %d Mode :%d \n", Sync, mode);
    
    struct sched_param high;
    memset( &high, 0, sizeof(high));
    high.sched_priority = 20;
    if(sched_setscheduler(getpid(), SCHED_FIFO, &high) == -1) perror("SCHED_FIFO :");
    // cpu affininty setting 
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
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
    
    char log_name[30];
    snprintf(log_name, 30, "./lalarand_log_%d.txt",getpid());
    freopen(log_name,"w",stderr);
    do{
        gpu_target = -1;
        cpu_target = -1;
        fd_head = make_fdset(&readfds, reg_fd, dnn_list);
        if(select(fd_head +1, &readfds, NULL, NULL, NULL)){
            current_time = get_time_point();
            // 1st registration check
            if(FD_ISSET(reg_fd, &readfds)) check_registration(dnn_list, reg_fd);
            
            // 2nd request check 
            for(node = dnn_list ->head; node !=NULL; node = node -> next) 
                if(FD_ISSET(node->request_fd, &readfds))
                    request_handler(node, gpu, cpu, profile_list[node->type], current_time);

            print_queue("GPU ",gpu->waiting);
            print_queue("CPU ",cpu->waiting);
            if(!(gpu->waiting->count + cpu->waiting->count < Sync)){
            
                if(Sync) update_deadline_all(dnn_list, current_time);

                if( gpu -> state == IDLE ) gpu_target = deQueue(gpu->waiting, dnn_list, profile_list, current_time, gpu);
                if( cpu -> state == IDLE ) cpu_target = deQueue(cpu->waiting, dnn_list, profile_list, current_time, cpu);
                
                if (mode == 4){ /* only in LaLaRAND */
                    if( gpu -> state == IDLE ) gpu_target = migration(cpu->waiting, dnn_list, profile_list, current_time, cpu, gpu);
                    if( cpu -> state == IDLE ) cpu_target = migration(gpu->waiting, dnn_list, profile_list, current_time, gpu, cpu);
                }
                
                if(gpu_target != -1) decision_handler(gpu_target, dnn_list, GPU);
                if(cpu_target != -1) decision_handler(cpu_target, dnn_list, CPU);
                Sync = 0;
            }
        }
    }while(!(Sync == 0 && dnn_list -> count == 0)); 
}   
