#define _GNU_SOURCE

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

#include "lalarand.h"
#include "lalarand_fn.h"

#define GPU 1
#define CPU 0

#define REGISTRATION "/tmp/lalarand_registration"

int main(int argc, char **argv){
    int Sync = find_int_arg(argc, argv, "-sync", 0);
    
    // cpu affininty setting 
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(1, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
    
    dnn_profile ** profile_list = make_profile_list();

    dnn_queue * dnn_list = createDNNQueue();

    resource * gpu = createResource();
    resource * cpu = createResource();

    // open channel for registration 
    int reg_fd = open_channel(REGISTRATION, O_RDONLY | O_NONBLOCK);
    
    double current_time;
    int request_layer, gpu_target, cpu_target;
    dnn_info * node;
    
    do{
        current_time = get_time_point();
    
        check_registration(dnn_list, reg_fd);

        // request handler
        if(check_request(dnn_list))
            for(node = dnn_list -> head; node != NULL ; node = node -> next)
                if(FD_ISSET(node -> request_fd, &readfds)) 
                    request_handler(node, gpu, cpu, profile_list[node->type], current_time);
                
        
        if(!(gpu->waiting->count + cpu->waiting->count < Sync)){
            
            if(Sync) update_deadline_all();

            if( gpu -> state == IDLE ) gpu_target = deQueue(gpu->waiting, dnn_list, profile_list, current_time, gpu);
            if( cpu -> state == IDLE ) cpu_target = deQueue(cpu->waiting, dnn_list, profile_list, current_time, gpu);

            // add migration policy

            if(gpu_target != -1) decision_handler(gpu_target, dnn_list, GPU);
            if(cpu_target != -1) decision_handler(cpu_target, dnn_list, CPU);
            
            Sync = 0;
        }


    }while(1)
}