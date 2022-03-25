#define DEBUG 0
#define debug_print(fmt, args...) \
            do { if (DEBUG) fprintf(stderr, fmt, ##args); fflush(stderr); } while (0)

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
#include <math.h>

#include "lalarand_fn.h"
#include "lalarand.h"
#define GPU 1
#define CPU 0

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b)  (((a) < (b)) ? (a) : (b))


void set_priority(int priority){
    struct sched_param prior;
    memset(&prior, 0, sizeof(prior));
    prior.sched_priority = priority;
    
    if(sched_setscheduler(getpid(), SCHED_FIFO, &prior) == -1) perror("SCHED_FIFO :");
}

void set_affinity(int core){
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(core, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
}

dnn_queue * createDNNQueue(){
    dnn_queue * tmp = (dnn_queue *)malloc(sizeof(dnn_queue));
    tmp -> count = 0;
    tmp -> head = NULL;
    return tmp;
}

void deleteNode(Queue * q, QNode * del){
    QNode * tmp, *prev;
    if(q->front == del){
        q->front = del->next;
        free(del);
        q->count --;
        return;
    }
    tmp = q->front;
    while( tmp !=del){
        prev = tmp;
        tmp = tmp->next;
    }
    prev->next = del->next;
    free(del);
    q->count --;
    return;
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

void setDNNpriority(dnn_queue * dnn_list){
    int len = dnn_list -> count;

    int smallest, pid;
    for(dnn_info * tmp = dnn_list->head; tmp != NULL; tmp = tmp->next)
        tmp->priority = -1;
    
    for(int i =1; i < len + 1; i++){
        smallest = 1000;
        pid = -1;
        for(dnn_info * tmp = dnn_list->head; tmp!=NULL; tmp = tmp->next){
            if(tmp->priority == -1){
                if(tmp->period < smallest) {
                    smallest = tmp->period;
                    pid = tmp->pid;
                }            
            }
        }
        for(dnn_info * tmp = dnn_list->head; tmp!=NULL; tmp = tmp->next)
            if (tmp -> pid == pid) tmp->priority = i;
    }

}

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

QNode* newNode (int layer, int id, int priority){
    QNode * tmp = (QNode *)malloc(sizeof(QNode));
    tmp -> layer = layer;
    tmp -> id = id;
    tmp -> next = NULL;
    tmp -> priority = priority;
    return tmp;
}

Queue * createQueue(){
    Queue * q = (Queue *)malloc(sizeof(Queue));
    q-> front =  NULL;
    q-> count = 0;
    q-> rear = NULL;
    return q;
}

void enQueue(Queue *q, int layer, int id, int priority){
    QNode * tmp = newNode(layer, id, priority);
    q->count ++;    
    
    debug_print("[enQueue] Enqueued [ID] %d [Layer] %d\n", id, layer);

    if(q->front == NULL){
        q->front = tmp;
        return;
    }    
    
    if(q->front -> priority > tmp-> priority ){
        tmp->next = q->front;
        q->front = tmp;
    }
    else{
        QNode * start = q->front;
        while(start->next != NULL && start->next->priority < tmp->priority){
            start = start->next;
        }

        tmp->next = start->next;
        start->next = tmp;
    }

}

int deQueue(Queue * q, double current_time, resource * res){  
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
    
    debug_print("[deQueue] [%s] Dequeued [ID] %d [layer] %d\n",get_resource_name(res->res_id), target_id, target_layer);

    return target_id;
}  

int find_node_by_id(Queue *q , int id){
    QNode * current = q->front;
    while(current !=NULL){
        if(current->id == id){
            return 1;
        }
        current = current -> next;
    }
    return 0;
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

void print_list(char * name, dnn_queue * dnn_list){
    dnn_info * head = dnn_list -> head;
    debug_print( "[Regist List] %s :", name);
    if (head == NULL) debug_print("Nothing registered");

    while( head != NULL){
        debug_print("{[%d] %d} ", head->id, head->priority);
        head = head->next;
    }
    debug_print("\n");
}


void print_queue(char * name, Queue * q){
    QNode * head =  q -> front;
    debug_print("[Queue List] %s :",name);
    if (head == NULL) 
        debug_print("Empty Queue"); 
  
    while (head != NULL) { 
        debug_print("{[%d] %d} ", head -> id, head -> layer);
        head = head->next; 
    }
    debug_print("\n");
}

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

void read_default_cfg(int pid, int * default_cfg){
    char *buffer = NULL;
    char *tmp;
    int size;
    int layer_num;
    
    FILE *fp = NULL;
    char filename[60];
    snprintf(filename, 60 , "/tmp/%d.txt", pid); 
    fp = fopen(filename, "r");
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    buffer = (char *)realloc(buffer,size + 1);

    fseek(fp, 0, SEEK_SET);
    fread(buffer, size, 1, fp);
    fclose(fp);
       
    tmp = strtok(buffer, ",");
    layer_num = atoi(tmp);
    for(int j= 0; j < layer_num; j++)
        default_cfg[j]=atoi(strtok(NULL,","));

    debug_print("Pid : %d ",pid);
    for(int i = 0; i < layer_num; i++){
        debug_print(",%d",default_cfg[i]);
    }
    debug_print("\n");
}

void check_registration(dnn_queue * dnn_list, int reg_fd, resource * gpu, resource *cpu){
    reg_msg * msg = (reg_msg *)malloc(sizeof(reg_msg));
        
    while( read(reg_fd, msg, 5*sizeof(int)) > 0){
        if(msg -> regist == 1) regist(dnn_list, msg); 
        else de_regist(dnn_list, msg, gpu, cpu);
    }
}

void regist(dnn_queue * dnn_list, reg_msg * msg){
    debug_print("REGISTRATION HAS BEEN DONE\n");
    dnn_info * dnn = (dnn_info *)malloc(sizeof(dnn_info));

    dnn -> id = dnn_list -> count;

    dnn -> pid = msg -> pid;
    dnn -> layers = msg -> layers;
    dnn -> period = msg -> period;
    dnn -> priority = msg -> priority;
    dnn -> current_layer = -1;
    dnn -> assigned = -1;
    dnn -> default_cfg = (int *)malloc(sizeof(int)* dnn->layers*2);

    read_default_cfg(dnn->pid, dnn->default_cfg);

    debug_print("======== REGISTRATION ========\n");
    debug_print("[ID]     %3d\n", dnn-> id);
    debug_print("[PID]    %3d\n", dnn-> pid);
    debug_print("[Layers] %3d\n", dnn-> layers);
    debug_print("[Prior]  %3d\n", dnn -> priority);
    debug_print("[Period] %3d\n", dnn->period);
    char req_fd_name[30];
    char dec_fd_name[30];

    snprintf(req_fd_name, 30,"/tmp/request_%d",dnn->pid);
    snprintf(dec_fd_name, 30,"/tmp/decision_%d",dnn->pid);

    dnn -> request_fd = open_channel(req_fd_name, O_RDONLY);
    dnn -> decision_fd = open_channel(dec_fd_name, O_WRONLY);
    
    dnn -> next = NULL;

    enDNNQueue(dnn_list, dnn);
    
}

void de_regist(dnn_queue * dnn_list, reg_msg * msg, resource * gpu, resource *cpu){
    int pid, target_id;
    dnn_info * target = find_dnn_by_pid(dnn_list, msg -> pid);
    target_id = target->id;
    pid = target -> pid;
    close_channels(target);
    deleteDNN(dnn_list, target);
     
    if (gpu-> id == target_id) gpu->state = IDLE;
    if (cpu-> id == target_id) cpu->state = IDLE;

    debug_print("================== %d DNN has been de registered ===================\n", pid); 
}

void request_handler(dnn_info * node, resource * gpu, resource * cpu, double current_time){
    
    req_msg msg;

    read(node -> request_fd, &msg, sizeof(int));
    int request_layer = msg.request_layer;
     
    debug_print("[Request handler] ID: %d , layer : %d  Time : %f\n", node->id, request_layer, current_time);
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
        node->current_layer = request_layer;
        debug_print("layer %d, resource %d\n",request_layer, node->default_cfg[request_layer]);
        if(node->default_cfg[request_layer] == GPU){
            enQueue(gpu->waiting, request_layer, node -> id, node -> priority);
        }else {
            enQueue(cpu->waiting, request_layer, node ->id, node -> priority);        
        }

    }
    else node->current_layer = -1;
}

void send_release_time(dnn_queue * dnn_list){
    struct timespec current_time;
    clock_gettime(CLOCK_MONOTONIC, &current_time);
    for(dnn_info * tmp = dnn_list-> head; tmp != NULL; tmp = tmp->next){
       if (write(tmp->decision_fd, &current_time.tv_sec, sizeof(current_time.tv_sec)) < 0)
            perror("release_time sec ");
       if (write(tmp->decision_fd, &current_time.tv_nsec, sizeof(current_time.tv_nsec)) < 0)
           perror("release_time nsec ");
    }
}

void decision_handler(int target_id, dnn_queue * dnn_list, int decision){
    cpu_set_t core;
    CPU_ZERO(&core);
    if(decision == GPU) CPU_SET(7, &core);
    if(decision == CPU) for(int i = 0; i < 7; i++) CPU_SET(i, &core);
    
    dnn_info * target = find_dnn_by_id(dnn_list, target_id);
    
    if(target->assigned != decision){
        sched_setaffinity(target->pid, sizeof(cpu_set_t), &core);
        target->assigned = decision;
    }

    if(decision == GPU) decision = 2;
    
    debug_print("[Decision handler] [ID] %d [Decision] %d\n", target_id, decision); 
   
    if( write(target->decision_fd,&decision,sizeof(int)) < 0){
        perror("decision_handler");  
    }
}

void update_deadline(dnn_info * dnn, double current_time){
    double micro_period = dnn->period * 1000;
    dnn-> deadline = current_time + micro_period;
    debug_print("Deadline update : [ID] %d ,[current] %f, [deadline] %f\n",dnn->id ,current_time, dnn -> deadline);
}

void update_deadline_all(dnn_queue * dnn_list, double current_time){
    for(dnn_info * node = dnn_list -> head ; node != NULL ; node = node -> next) update_deadline(node, current_time);
}

char * get_resource_name(int id){
    if(id == GPU) return "GPU";
    if(id == CPU) return "CPU";
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
        debug_print("[ERROR]Fail to open channel for %s\n", pipe_name);
        exit(-1);
    }
   debug_print("Channel for %s has been successfully openned!\n", pipe_name);
   
   return pipe_fd;
}

void close_channel(char * pipe_name){
    if ( unlink(pipe_name) == -1){
        debug_print("[ERROR]Fail to remove %s\n",pipe_name);
        exit(-1);
    }
}

void close_channels(dnn_info * dnn){
    char request_name[30];
    char decision_name[30];
    
    snprintf(request_name, 30, "/tmp/request_%d", dnn->pid);
    snprintf(decision_name, 30, "/tmp/decision_%d", dnn->pid);
    
    close_channel(request_name);
    close_channel(decision_name);
    
}

int make_fdset(fd_set *readfds,int reg_fd, dnn_queue * dnn_list){
    FD_ZERO(readfds);

    FD_SET(reg_fd, readfds);
        
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

