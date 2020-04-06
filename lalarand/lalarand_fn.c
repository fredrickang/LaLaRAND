#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <signal.h>
#include <time.h>
#include <chrono>
#include <float.h>

#include "lalarand_fn.h"
#define GPU 1
#define CPU 0
#define GPU_MAX 1069


///// dnn queue ////

dnn_queue * createDNNQueue(){
    dnn_queue * tmp = (dnn_queue *)malloc(sizeof(dnn_queue));
    tmp -> count = 0;
    tmp -> head = NULL;

    return tmp;
}


void deleteDNN(dnn_queue * dnn_list, dnn_info * del){ 
    dnn_info * tmp , *prev;
    
    if(dnn_list -> head == del){
        dnn_list -> head = del -> next;
        free(del);
        dnn_list -> count --;
        return;
    }
    tmp = dnn_list -> head;
    while( tmp != del){
        prev = tmp;
        tmp = tmp -> next;
    }
    prev -> next  = del -> next;
    free(del);
    dnn_list -> count --;
    return;
} 

void enDNNQueue(dnn_queue * dnn_list, dnn_info * dnn){
    if(dnn_list->head == NULL){
        dnn_list -> head = dnn;
        dnn_list -> count ++;
        return;
    }    

    dnn -> next = dnn_list -> head;
    dnn_list -> head = dnn;
    dnn_list -> count ++ ; 

}

///// resource /////

resource * createResource(int res_id){
    resource * tmp = (resource *)malloc(sizeof(resource));
    tmp -> state = IDLE;
    tmp -> id = -1;
    tmp -> res_id = res_id;
    tmp -> layer = -1;
    tmp -> scheduled = -1;
    tmp -> waiting = createQueue();
    return tmp;
}


//// waiting queue ////

QNode* newNode (int layer, int id, int period){
    QNode * tmp = (QNode *)malloc(sizeof(QNode));
    tmp -> layer = layer;
    tmp -> id = id;
    tmp -> next = NULL;
    tmp -> prev = NULL;
    tmp -> period = period;

    return tmp;

}

Queue * createQueue(){

    Queue * q = (Queue *)malloc(sizeof(Queue));
    q-> front =  NULL;
    q-> count = 0;
    return q;
    
}

void enQueue(Queue *q, int layer, int id, int period){
    QNode * tmp = newNode(layer, id, period);
    q->count ++;    
    
    if(q->front == NULL){
        q->front = tmp;
        fprintf(stderr,"Enqueue : [ID] %d [layer] %d \n", id, layer);
        return;
    }    

    fprintf(stderr,"Enqueue : [ID] %d [layer] %d \n", id, layer);
    
    if(q->front -> period > tmp-> period ){
        tmp->next = q->front;
        q->front = tmp->next;
    }

    else{
        QNode * start = q->front;
        while(start->next != NULL && start->next->period > tmp->period){
            start = start->next;
        }

        tmp->next = start->next;
        start->next = tmp;
    }

}

int deQueue(Queue * q, dnn_queue * dnn_list, dnn_profile ** profile_list, double current_time, resource * res){
    // if there is nothing to de queue
    if (q -> front == NULL){
        return -1;
    }

    QNode * target = q->front;
    q->front = target->next;
    
    int target_id = target->id;
    int target_layer = target->layer;

    res -> state = BUSY;
    res -> id  = target_id;  
    res -> layer = target_layer;
    res -> scheduled = current_time;
    
    q -> count --;
    free(target);
    return target_id;
}  

dnn_info * find_dnn_by_id(dnn_queue * dnn_list, int id){
    dnn_info * node = dnn_list -> head;
    while(node -> id != id){
        node =  node -> next;
    }
    return node;
}

dnn_info * find_dnn_by_pid(dnn_queue * dnn_list, int pid){
    dnn_info * node = dnn_list -> head;
    while(node -> pid != pid){
        node = node -> next;
    }
    return node;
}   


void print_queue(char * name, Queue * q){
    QNode * head =  q -> front;
    fprintf(stderr,"%s :",name);
    if (head == NULL) 
        fprintf(stderr,"Doubly Linked list empty"); 
  
    while (head != NULL) { 
        fprintf(stderr,"{[%d] %d} ", head -> id, head -> layer);
        head = head->next; 
    }
    puts("");
}

///// Utils ////
void del_arg(int argc, char **argv, int index)
{
    int i;
    for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
    argv[i] = 0;
}

int find_int_arg(int argc, char **argv, char *arg, int def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atoi(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

double get_time_point(){
    std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(current_time.time_since_epoch()).count();
}


//// LaLaRAND ////
void make_profile(dnn_profile * tmp, int layers, int *gpu, int *cpu, int *cfg){
    
    tmp->gpu_exec = (int *)malloc(sizeof(int) * layers);
    tmp->cpu_exec = (int *)malloc(sizeof(int) * layers);
    tmp->cfg = (int *)malloc(sizeof(int) * layers);

    memcpy(tmp->gpu_exec, gpu, sizeof(int) *layers);
    memcpy(tmp->cpu_exec, cpu, sizeof(int) *layers);
    memcpy(tmp->cfg , cfg, sizeof(int) * layers);
}

dnn_profile ** make_profile_list(int mode){
    // mode 1: ALL GPU // mode 2: preferable // mode 3: Static //mode 4: LaLaRAND

    dnn_profile ** profile_list = (dnn_profile **)malloc(sizeof(dnn_profile *)*4);

    for(int i =0; i < 4; i++)
        profile_list[i] = (dnn_profile *)malloc(sizeof(dnn_profile));

    

    int yolo_gpu[24]  = {1069 ,152 ,462 ,125 ,317 ,113 ,262 ,87 ,256 ,99 ,275 ,104 ,735 ,177 ,273 ,135 ,110 ,81 ,124 ,103 ,92 ,466 ,157 ,122 };
    int yolo_cpu[24] = {7240 ,1476 ,12392 ,608 ,15986 ,255 ,8566 ,121 ,8661 ,65 ,11764 ,414 ,46709 ,2596 ,11455 ,1387 ,94 ,20 ,375 ,180 ,126 ,27672 ,1844 ,302 };
    int yolo_cfg[24] = {1 ,1 ,1, 1 ,1 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,1 ,1 ,1 }; 

    int extraction_gpu[28] = {360,111,361,107,146,255,199,608,93,153,281,146,278,146,279,148,278,184,740,85,155,530,153,535,177,103,265,6};
    int extraction_cpu[28] = {12051 ,704 ,16830 ,498 ,963 ,11076 ,2466 ,46710 ,327 ,1904 ,13246 ,1534 ,13249 ,1537 ,13493 ,1512 ,13091 ,2953 ,51395 ,172 ,3623 ,26663 ,2888 ,25415 ,5559 ,23 ,10 ,1 };
    int extraction_cfg[28]=  {1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1, 1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,0 ,0 };
    
    
    int resnet_gpu[29] = {429 ,114 ,213 ,196 ,102 ,211 ,197 ,117 ,203 ,261 ,110 ,268 ,264 ,105 ,198 ,191 ,105 ,197 ,191 ,100 ,202 ,318 ,88 ,322 ,318 ,101 ,108 ,129 ,253 };
    int resnet_cpu[29] = {14555,828,9231,9147,156,9274,9314,159,3637,7197,167,7355,7356,138,3543,7009,71,7071,7045,72,4846,9627,33,9515,9394,15,14,732,14};
    int resnet_cfg[29] =  {1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ,0 ,1 ,1 ,0 ,1 ,1 ,0 ,0 ,1 ,0 };
    

    int rnn_gpu[6] = {208, 223, 216, 53, 51, 2};
    int rnn_cpu[6] = {279, 334, 336, 30, 7, 1};
    int rnn_cfg[6] = {1,1,1,0,0,0};
    
    for(int i = 0; i < 24; i++){
        yolo_gpu[i] = yolo_gpu[i]*10;
        yolo_cpu[i] = yolo_cpu[i]*10;
    }
    for(int i =0; i < 28; i++){
        extraction_gpu[i] = extraction_gpu[i]* 10;
        extraction_cpu[i] = extraction_cpu[i]* 10;
    }
    for(int i =0; i < 29; i++){
        resnet_gpu[i] = resnet_gpu[i] * 10;
        resnet_cpu[i] = resnet_cpu[i] * 10;
    }
    for(int i =0; i < 6; i++){
        rnn_gpu[i] = rnn_gpu[i] * 10;
        rnn_cpu[i] = rnn_cpu[i] * 10;
    }


    if (mode == 1){
        memset(yolo_cfg,1,sizeof(int)*24);
        memset(extraction_cfg,1, sizeof(int)*28);
        memset(resnet_cfg,1, sizeof(int)*29);
        memset(rnn_cfg,1, sizeof(int)*6);
    }
    if (mode == 3){
        memset(yolo_cfg, 0, sizeof(int)*24);
        memset(extraction_cfg, 0, sizeof(int)*28);
        memset(resnet_cfg, 0, sizeof(int)*29);
        memset(rnn_cfg, 0, sizeof(int)*6);

        memset(yolo_cfg, 1, sizeof(int)*12);
        memset(extraction_cfg, 1, sizeof(int)*14);
        memset(resnet_cfg, 1, sizeof(int)*14);
        memset(rnn_cfg, 1, sizeof(int)*3);
    }

    
    make_profile(profile_list[YOLOt], 24, yolo_gpu, yolo_cpu, yolo_cfg);
    make_profile(profile_list[EXTRACTION], 28, extraction_gpu, extraction_cpu, extraction_cfg);
    make_profile(profile_list[RESNET], 29, resnet_gpu, resnet_cpu, resnet_cfg);
    make_profile(profile_list[RECURRENT], 5, rnn_gpu, rnn_cpu, rnn_cfg);

    return profile_list;
}

void check_registration(dnn_queue * dnn_list, int reg_fd){
    reg_msg * msg = (reg_msg *)malloc(sizeof(reg_msg));
        
    while( read(reg_fd, msg, 5*sizeof(int)) > 0){
        if(msg -> regist == 1) regist(dnn_list, msg); 
        else de_regist(dnn_list, msg);
    }
}

void regist(dnn_queue * dnn_list, reg_msg * msg){
    dnn_info * dnn = (dnn_info *)malloc(sizeof(dnn_info));

    dnn -> id = dnn_list -> count;

    dnn -> pid = msg -> pid;
    dnn -> layers = msg -> layers;
    dnn -> type = msg -> type;
    dnn -> period = msg -> period;
    
    printf("======== REGISTRATION ========\n");
    printf("[ID]     %3d\n", dnn-> id);
    printf("[PID]    %3d\n", dnn-> pid);
    printf("[Layers] %3d\n", dnn-> layers);
    printf("[Type]   %s\n", get_dnn_name(dnn->type));
    printf("[Period] %3d\n", dnn->period);
    char req_fd_name[30];
    char dec_fd_name[30];

    snprintf(req_fd_name, 30,"/tmp/request_%d",dnn->pid);
    snprintf(dec_fd_name, 30,"/tmp/decision_%d",dnn->pid);

    dnn -> request_fd = open_channel(req_fd_name, O_RDONLY);
    dnn -> decision_fd = open_channel(dec_fd_name, O_WRONLY);
    
    dnn -> next = NULL;

    enDNNQueue(dnn_list, dnn);
    
}

void de_regist(dnn_queue * dnn_list, reg_msg * msg){
    // find dnn info by pid;
    // delete dnn 
    // send signal to go 
    int pid;
    dnn_info * target = find_dnn_by_pid(dnn_list, msg -> pid);
    pid = target -> pid;
    deleteDNN(dnn_list, target);
    printf("%d DNN has been de registered\n", pid); 
}

int check_request(dnn_queue * dnn_list, fd_set* readfds, int sync){
    int rev = 0;
    struct timeval zero = {0, 0};
    sigset_t set;
    int signum;
    if(dnn_list -> count >= sync && dnn_list -> count != 0){
        FD_ZERO(readfds);
    
        dnn_info * node = dnn_list -> head;
        while(node != NULL){
            FD_SET(node -> request_fd, readfds);
            node = node -> next;
        }

        rev = select(dnn_list -> head -> request_fd +1, readfds, NULL, NULL, &zero);
        sigemptyset(&set);
        sigaddset(&set, SIGCONT);
        sigprocmask(SIG_SETMASK, &set, NULL);
        if( rev == 0 ) sigwait(&set, &signum);
    }
    return rev;
}

void request_handler(dnn_info * node, resource * gpu, resource * cpu, dnn_profile * profile, double current_time){
    
    int request_layer;
    
    read(node -> request_fd, &request_layer, sizeof(int));
    
    //fprintf(stderr,"[request_handler] : [ID] %d [layer] %d \n", node -> id, request_layer);

    if(request_layer == 0) update_deadline(node, current_time);

    if( gpu -> state == BUSY && gpu -> id == node->id){
        gpu -> state = IDLE;
        gpu -> id = -1;
        gpu -> layer = -1;
        gpu -> scheduled = -1;
      }

    if( cpu -> state == BUSY && cpu -> id == node->id){
        cpu -> state = IDLE;
        cpu -> id = -1;
        cpu -> layer  = -1;
        cpu -> scheduled = -1;
      }

     if(request_layer != node -> layers){
         if(profile->cfg[request_layer] == GPU) enQueue(gpu->waiting,request_layer, node ->  id, node -> period);
         else enQueue(cpu->waiting, request_layer, node -> id, node -> period );
      }
}


void decision_handler(int target_id, dnn_queue * dnn_list, int decision){
    int rev;
    dnn_info * target = find_dnn_by_id(dnn_list, target_id);
    if( write(target->decision_fd,&decision,sizeof(int)) < 0){
        perror("decision_handler");  
    }
}

void update_deadline(dnn_info * dnn, double current_time){
    // milli period to micro period
    double micro_period = dnn->period * 1000;
    dnn-> deadline = current_time + micro_period;
    fprintf(stderr,"Deadline update : [dnn] %s ,[current] %f, [deadline] %f\n", get_dnn_name(dnn->type), current_time, dnn -> deadline);
}

void update_deadline_all(dnn_queue * dnn_list, double current_time){
    for(dnn_info * node = dnn_list -> head ; node != NULL ; node = node -> next) update_deadline(node, current_time);
}

char* get_dnn_name(DNN_TYPE type){
    if(type == 0) return "YOLO";
    if(type == 1) return "EXTRACTION";
    if(type == 2) return "RESNET";
    if(type == 3) return "RNN";
}


double workload_left(dnn_profile * profile, int current_layer, int layer_num){
    int workload = 0;
    
    for(int i = current_layer; i < layer_num; i++)
        workload += (profile->cfg[i] == 0) ? profile->cpu_exec[i] : profile->gpu_exec[i];
    
    return workload;
}

int migration(Queue * q, dnn_queue * dnn_list, dnn_profile ** profile_list, double current_time, resource * From, resource * To){
    
    if(q->count == 0)
        return -1;
    
    double slack, future_wait, blocked;
    dnn_info * node;
    int target_id = -1;
    int target_layer = -1;
    double smallest = DBL_MAX;

    for(QNode * tmp = q->front; tmp != NULL; tmp = tmp -> next){
        node = find_dnn_by_id(dnn_list, tmp -> id);
        slack = node->deadline - current_time - workload_left(profile_list[node->type],tmp -> layer, node->layers);
        if( slack > abs(profile_list[node->type]->gpu_exec[tmp->layer] - profile_list[node->type]->cpu_exec[tmp->layer])) { /* first condidtion */
            future_wait = waiting(q, dnn_list, profile_list, current_time, From, tmp->id);
            if ( future_wait - GPU_MAX > abs(profile_list[node->type]->gpu_exec[tmp->layer] - profile_list[node->type]->cpu_exec[tmp->layer]))
                if( slack < smallest ){
                    target_id = tmp -> id;
                    target_layer = tmp -> layer;
                    smallest = slack;
                }
        }
    }

    if(target_id != -1){
        To -> state = BUSY;
        To -> id = target_id;
        To -> layer = target_layer;
        To -> scheduled = current_time;
        
        QNode * tmp = q->front;
        QNode * prev;
        if (tmp ->id == target_id){
            q->front = q->front->next;
            free(tmp);
        }
        else{
            while(tmp != NULL && tmp->id != target_id){
                prev = tmp;
                tmp = tmp->next;
            }

            prev->next = tmp->next;
            free(tmp);
        }
    
        q -> count --;

        fprintf(stderr,"Migration : [ID] %d [Layer] %d \n",target_id, target_layer);
    }

    return target_id; 
}

double waiting(Queue * q, dnn_queue * dnn_list, dnn_profile ** profile_list, double current_time, resource * From, int target_id){
    double waited = 0;

    dnn_info * current = find_dnn_by_id(dnn_list, From->id);
    dnn_info * target = find_dnn_by_id(dnn_list, target_id);
    double current_wait = From->res_id == GPU ? profile_list[current->type]->gpu_exec[From->layer] - (current_time - From->scheduled) : profile_list[current->type]->cpu_exec[From->layer] - (current_time - From->scheduled);

    waited += current_wait;

    if(target->period > current->period){
        for(int i = From->layer+1 ; i < current->layers ; i++){
            if (profile_list[current->type]->cfg[i] == From->res_id) waited += From->res_id == GPU ? profile_list[current->type]->gpu_exec[i] : profile_list[current->type]->cpu_exec[i];
            else break;
        }
    }
    
    for(QNode *tmp = q->front; tmp->period < target->period; tmp = tmp->next){
        dnn_info * dnn = find_dnn_by_id(tmp->id);
        for(int i = tmp->layer; i < dnn->layers ; i ++){
            if(profile_list[dnn->type]->cfg[i] == From->res_id) waited += From->res_id == GPU ? profile_list[current->type]->gpu_exec[i] : profile_list[current->type]->cpu_exec[i];
            else break;
        }
    }

    return waited;
}

///// communication ////

int open_channel(char * pipe_name,int mode){
    int pipe_fd;
    
    if( access(pipe_name, F_OK) != -1)
        remove(pipe_name);

    if( mkfifo(pipe_name, 0666) == -1){
        puts("[ERROR]Fail to make pipe");
        exit(-1);
    }
    if( (pipe_fd = open(pipe_name, mode)) < 0){
        printf("[ERROR]Fail to open channel for %s\n", pipe_name);
        exit(-1);
    }
   printf("Channel for %s has been successfully openned!\n", pipe_name);
   
   return pipe_fd;
}

void close_channel(char * pipe_name){
    if ( unlink(pipe_name) == -1){
        printf("[ERROR]Fail to remove %s\n",pipe_name);
        exit(-1);
    }
}

void close_channels(dnn_info ** dnn_list, int dnns){
    char request_name[30];
    char decision_name[30];
    for(int i = 0; i < dnns; i++){
        snprintf(request_name, 30, "lalarand_request_%d", dnn_list[i]->pid);
        snprintf(decision_name, 30, "lalarand_decision_%d", dnn_list[i]->pid);
        close_channel(request_name);
        close_channel(decision_name);
    }
}

int make_fdset(fd_set *readfds,int reg_fd, dnn_queue * dnn_list){
    // initialize fd_set;
    FD_ZERO(readfds);

    // set register_fd
    FD_SET(reg_fd, readfds);
        
    // if there exist registered dnn, set
    if(dnn_list -> count > 0){
        dnn_info * node = dnn_list -> head;
        while(node != NULL){
            FD_SET(node -> request_fd, readfds);
            node = node -> next;
        }
        return dnn_list -> head ->request_fd;
    }
    return reg_fd;
}






