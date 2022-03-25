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

    set_priority(50); 
    set_affinity(7);
    
    dnn_queue * dnn_list = createDNNQueue();
    resource * gpu = createResource(GPU);
    resource * cpu = createResource(CPU);
    
    int reg_fd = open_channel(REGISTRATION, O_RDONLY | O_NONBLOCK);
    
    double current_time;
    int gpu_target, cpu_target;
    int fd_head;
    fd_set readfds;
    dnn_info *node;
    
    do{
        gpu_target = -1;
        cpu_target = -1;

        fd_head = make_fdset(&readfds, reg_fd, dnn_list);
    
        if(select(fd_head +1, &readfds, NULL, NULL, NULL)){
            current_time = get_time_point();
            if(FD_ISSET(reg_fd, &readfds)) {
                check_registration(dnn_list, reg_fd, gpu, cpu);
                print_list("REGIST",dnn_list);
            }
            for(node = dnn_list ->head; node !=NULL; node = node -> next){
                if(FD_ISSET(node->request_fd, &readfds)){
                    request_handler(node, gpu, cpu, current_time);
                }
            }
            
            print_queue("GPU",gpu->waiting);
            print_queue("CPU",cpu->waiting);
            
            if(!(gpu->waiting->count + cpu->waiting->count < Sync)){
                if(Sync) update_deadline_all(dnn_list, current_time);

                if(gpu -> state == IDLE) gpu_target = deQueue(gpu->waiting, current_time, gpu);
                if(cpu -> state == IDLE) cpu_target = deQueue(cpu->waiting, current_time, cpu);
            
                if(Sync) send_release_time(dnn_list);
               
                if(gpu_target != -1) decision_handler(gpu_target, dnn_list, GPU);
                if(cpu_target != -1) decision_handler(cpu_target, dnn_list, CPU);
                Sync = 0;
            }
        }
    }while(!(Sync == 0 && dnn_list -> count == 0)); 
}
