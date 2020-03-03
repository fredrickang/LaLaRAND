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

#include "lalarand.h"
#include "lalarand_fn.h"

#define GPU 1
#define CPU 0

#define REGISTRATION "/tmp/lalarand_registration"

int main(int argc, char **argv){
    int Sync = find_int_arg(argc, argv, "-sync", 1);
    
    // cpu affininty setting 
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
    
    dnn_profile ** profile_list = make_profile_list();

    dnn_queue * dnn_list = createDNNQueue();

    resource * gpu = createResource();
    resource * cpu = createResource();

    int reg_fd = open_channel(REGISTRATION, O_RDONLY | O_NONBLOCK);
    
    double current_time;
    int gpu_target, cpu_target;
    dnn_info * node;
    fd_set readfds; 
    
    do{
        gpu_target =  -1;
        cpu_target =  -1;
        current_time = get_time_point();
    
        check_registration(dnn_list, reg_fd);
        
        if(check_request(dnn_list, &readfds, Sync))
            for(node = dnn_list -> head; node != NULL; node = node -> next)
                if(FD_ISSET(node -> request_fd, &readfds))
                    request_handler(node, gpu, cpu, profile_list[node->type], current_time);       
        
        if(!(gpu->waiting->count + cpu->waiting->count < Sync)){
            
            if(Sync) update_deadline_all(dnn_list, current_time);

            if( gpu -> state == IDLE ) gpu_target = deQueue(gpu->waiting, dnn_list, profile_list, current_time, gpu);
            if( cpu -> state == IDLE ) cpu_target = deQueue(cpu->waiting, dnn_list, profile_list, current_time, cpu);

            if( gpu -> state == IDLE ) gpu_target = migration(cpu->waiting, dnn_list, profile_list, current_time, gpu);
            if( cpu -> state == IDLE ) cpu_target = migration(gpu->waiting, dnn_list, profile_list, current_time, cpu);

            if(gpu_target != -1) decision_handler(gpu_target, dnn_list, GPU);
            if(cpu_target != -1) decision_handler(cpu_target, dnn_list, CPU);
            
            Sync = 0;
            
           
        }
    }while(!(Sync == 0 && dnn_list -> count == 0));
}
