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

int main(int argc, char **argv){
    //Input argument handling
    int dnns = find_int_arg(argc, argv, "-dnns", 1);
    int Sync = find_int_arg(argc, argv, "-sync", 1);
    
    //CPU affinity to core 1
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(1, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
    
    int request_fd[dnns];    
    int decision_fd[dnns];
    //Registeration part
    register_fd = open_channel("./lalarand_register",O_RDWR);
    
    
    char request_name[30];
    for(int i =0; i < dnns; i++){
        snprintf(request_name, 30, "lalarand_request_%d", dnn_list[i]->pid);
        request_fd[i] = open_channel(request_name, O_RDONLY); 
    }
    puts("\nrequest channel has been openned\n");
    
    //Open Decision channel
    
    char decision_name[30];
    for(int i =0; i < dnns; i++){
        snprintf(decision_name, 30, "lalarand_decision_%d", dnn_list[i]->pid);
        decision_fd[i] = open_channel(decision_name, O_WRONLY);
    }
    puts("\ndecision channel has been openned\n"); 
    dnn_profile ** profile_list = make_profile_list();
    int request_layer = -1;
    int go_to_gpu = 1;
    int go_to_cpu = 0;
    int err;
   
    Queue * gpu_request = createQueue("GPU_REQUEST");
    Queue * cpu_request = createQueue("CPU_REQEUST");
    resource* gpu = (resource *)malloc(sizeof(resource));
    resource* cpu = (resource *)malloc(sizeof(resource)); 
    
    gpu -> state = IDLE;
    cpu -> state = IDLE;

    gpu -> id = -1;
    cpu -> id = -1;
    
    int gpu_target_id;
    int cpu_target_id;
    
    
    fd_set readfds, writefds;
    int state;
    double current_time = 0;
    struct timeval zero;
    zero.tv_sec = 0;
    zero.tv_usec = 0;

    while(1){
        
        current_time = get_time_point();
        FD_ZERO(&readfds); 
        for(int i = 0 ; i < dnns; i++)
            FD_SET(request_fd[i], &readfds);
        

        // REQUEST handler 
        state = select(request_fd[dnns-1] + 1, &readfds, NULL, NULL, &zero);
        switch(state){
            case -1:
                perror("select error : ");
                exit(-1);
                break;
            
            case 0:
                break;
            default:
                for(int i =0; i < dnns; i ++){
                    if (FD_ISSET(request_fd[i], &readfds)){
                        
                        if( read(request_fd[i], &request_layer, sizeof(int)) < 0){
                            perror("read error : ");    
                            exit(-1);
                        }
                        
                        if(request_layer == 0) update_deadline(dnn_list[i], current_time);
                        
                        if( gpu -> state == BUSY && gpu -> id == i){
                            gpu -> state = IDLE;
                            gpu -> id = -1;
                        }

                        if( cpu -> state == BUSY && cpu -> id == i){
                            cpu -> state = IDLE;
                            cpu -> id = -1;
                        }
                        
                        if(request_layer != dnn_list[i]->layers){
                            printf("Request : [ID] %d [layer] %d \n", i, request_layer);
                            if(profile_list[dnn_list[i]->type]->cfg[request_layer] == GPU) enQueue(gpu_request, request_layer, i);
                            else enQueue(cpu_request, request_layer, i); 
                        }
                    }
                }
        }
                
        // Decision 
        if(!(Sync && (gpu_request->count + cpu_request->count) < dnns)){
            cpu_target_id = -1;
            gpu_target_id = -1;

            if(Sync) for(int i =0 ; i < dnns ; i++) update_deadline(dnn_list[i], current_time);
            
            if(gpu -> state == IDLE) gpu_target_id = deQueue(gpu_request, dnn_list, profile_list ,current_time, gpu);    
            if(cpu -> state == IDLE) cpu_target_id = deQueue(cpu_request, dnn_list, profile_list ,current_time, cpu);
            else{ /* preemption */
                // need to be implemented 
            }
           
            //if( !(gpu_target_id == -1 && cpu_target_id == -1) || (gpu_target != -1 && cpu_target_id != -1) ){ /* migration condition 1 & 2 */

            //}

            
            if(gpu_target_id != -1){
                if(write(decision_fd[gpu_target_id], &go_to_gpu, sizeof(int)) < 0){
                    perror("gpu decision : ");
                    exit(-1);
                }
            } 
            
            if(cpu_target_id != -1){
                if(write(decision_fd[cpu_target_id], &go_to_cpu, sizeof(int)) < 0){
                    perror("cpu decision :");
                    exit(-1);
                }
            }

            Sync = 0;
        }
        
    }
}
