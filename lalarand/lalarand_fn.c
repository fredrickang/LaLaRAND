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

#include "lalarand_fn.h"
#define GPU 1
#define CPU 0


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
    return q;
    
}

void enQueue(Queue *q, int layer, int id, int priority){
    QNode * tmp = newNode(layer, id, priority);
    q->count ++;    
    
    if(q->front == NULL){
        q->front = tmp;
        debug_print("Enqueue : [ID] %d [layer] %d \n", id, layer);
        return;
    }    

    debug_print("Enqueue : [ID] %d [layer] %d \n", id, layer);
    
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

int deQueue_algo(Queue *q, dnn_queue * dnn_list, dnn_profile ** profile_list, double current_time, resource * res){
    // if there is nothing to de queue
    if (q -> front == NULL){
        return -1;
    }
    QNode * RR = NULL;
    for(QNode * tmp = q->front; tmp != NULL; tmp = tmp->next){
        if (tmp -> layer == 0){
              RR = tmp;
              break;
        }
    }


    QNode * tmp = q->front;
    if(RR != NULL){
        QNode * prev;         
        if (tmp ->id == RR->id)
            q->front = q->front->next;    
        else{
            while(tmp != NULL && tmp->id != RR->id){
                prev = tmp;
                tmp = tmp->next;
            }

            prev->next = tmp->next;
        }
    }else{
        q->front = tmp->next;
    }


    int target_id = tmp->id;
    int target_layer = tmp->layer;

    res -> state = BUSY;
    res -> id  = target_id;  
    res -> layer = target_layer;
    res -> scheduled = current_time;
    
    q -> count --;
    free(tmp);
    debug_print("Dequeue : [ID] %d [layer] %d \n", target_id, target_layer);
    return target_id;

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
    debug_print("Dequeue : [ID] %d [layer] %d \n", target_id, target_layer);
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

void print_list(char * name, dnn_queue * dnn_list){
    dnn_info * head = dnn_list -> head;
    debug_print( "%s :", name);
    if (head == NULL) debug_print("Nothing registered");

    while( head != NULL){
        debug_print("{[%d] %d} ", head->id, head->priority);
        head = head->next;
    }
    debug_print("\n");
}


void print_queue(char * name, Queue * q){
    QNode * head =  q -> front;
    debug_print("%s :",name);
    if (head == NULL) 
        debug_print("Empty Queue"); 
  
    while (head != NULL) { 
        debug_print("{[%d] %d} ", head -> id, head -> layer);
        head = head->next; 
    }
    debug_print("\n");
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
void make_profile(dnn_profile * tmp, int layers, int *gpu, int *cpu, int *cfg, int *G2C, int * C2G){
    
    tmp->gpu_exec = (int *)malloc(sizeof(int) * layers);
    tmp->cpu_exec = (int *)malloc(sizeof(int) * layers);
    tmp->cfg = (int *)malloc(sizeof(int) * layers);
    tmp->G2C = (int *)malloc(sizeof(int)* (layers-1));
    tmp->C2G = (int *)malloc(sizeof(int)* (layers-1));

    memcpy(tmp->gpu_exec, gpu, sizeof(int) *layers);
    memcpy(tmp->cpu_exec, cpu, sizeof(int) *layers);
    memcpy(tmp->cfg , cfg, sizeof(int) * layers);
    memcpy(tmp->G2C, G2C, sizeof(int) * (layers-1));
    memcpy(tmp->C2G, C2G, sizeof(int) * (layers-1));
}

dnn_profile ** make_profile_list(int baseline){
    dnn_profile ** profile_list = (dnn_profile **)malloc(sizeof(dnn_profile *)*4);

    for(int i =0; i < 4; i++)
        profile_list[i] = (dnn_profile *)malloc(sizeof(dnn_profile));


    int yolo_gpu[24]  = {305,  37, 124,  21,  74,  17,  59,  14,  52,   9,  60,  12, 182, 39,  55,  27,  44,   5,  31,  14,  12, 101,  31,  48};
    int yolo_cpu[24] = { 3287,   698,  6423,   352,  9595,   323,  6334,   175,  6590, 54,  8118,   104, 32054,  1894,  8062,   967,   101,    11, 366,    70,    54, 19150,  1530,   388};
    int yolo_cfg[24] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }; 
    int yolo_data_G2C[23] = {3175,1766,1228,716,880,573,675,563,561,594,589,548,610,562,563,516,151,499,487,513,627,547,610}; 
    int yolo_data_C2G[23] = {1709,554,904,288,543,540,306,386,569,363,494,452,524,105,472,100,72,113,79,352,488,542,493};


    int extraction_gpu[28] = {94,  28,  84,  15,  37,  53,  40, 144,  13,  30,  57,  30,  56, 30,  56,  31,  56,  37, 184,  11,  28, 132,  29, 132,  35,   9,50,   2}; 
    int extraction_cpu[28] = {7068,   210, 10416,   162,   727,  7156,  1645, 28586,   113, 1020,  7878,  1011,  7860,  1009,  7853,  1009,  7860,  1864, 30857,    62,  1816, 15826,  1797, 15833,  3493,    14,     6, 1};
    int extraction_cfg[28]=  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 };
    int extraction_data_G2C[27] = {2038,1029,623,529,417,534,327,986,664,337,463,188,351,200,274,194,252,296,340,268,150,230,327,220,348,207,301};
    int extraction_data_C2G[27] = {627,488,469,453,161,482,522,425,375,136,386,129,468,118,435,100,419,396,560,106,96,105,112,132,101,407,394};

    int resnet_gpu[29] = {113,  30,  47,  40,  18,  43,  47,  16,  41,  57,  15,  57,  54, 11,  36,  36,  13,  35,  34,   9,  44,  70,  16,  72,  75,  10, 13,  26,  49};
    int resnet_cpu[29] = {7398,  271, 4892, 4752,   65, 4770, 4726,   65, 2460, 4914,   86, 4954, 4949,   35, 2609, 5131,   40, 5116, 5102,   22, 3702, 7235, 25, 7283, 7302, 12, 11, 374, 6};
        
    int resnet_cfg[29] =  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0};
    int resnet_data_G2C[28] = {2112,1228,566,650,612,624,628,697,660,541,544,618,482,535,421,436,518,404,537,594,536,565,473,483,514,474,132,1};
    int resnet_data_C2G[28] = {870,517,548,584,546,556,542,542,491,500,459,518,544,306,356,345,220,386,341,339,112,149,87,110,131,115,226,1};

    int rnn_gpu[6] = {130, 113, 107,  28,  20,   3};
    int rnn_cpu[6] = {113, 139, 139,  16,   5,   4};
    int rnn_cfg[6] = {0, 1, 1, 0, 0, 1};
    int rnn_data_G2C[5] = {1518,452,573,178,426};
    int rnn_data_C2G[5] = {308,374,355,236,236};

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

    if (baseline == 1 || baseline == 3){
        for(int i = 0; i < 24; i++) yolo_cfg[i] = 1;
        for(int i = 0; i < 28; i++) extraction_cfg[i] = 1;
        for(int i = 0; i < 29; i++) resnet_cfg[i] = 1;
        for(int i = 0; i < 6 ; i++) rnn_cfg[i] = 1;
    }
    
    make_profile(profile_list[YOLOt], 24, yolo_gpu, yolo_cpu, yolo_cfg, yolo_data_G2C, yolo_data_C2G);
    make_profile(profile_list[EXTRACTION], 28, extraction_gpu, extraction_cpu, extraction_cfg, extraction_data_G2C, extraction_data_C2G);
    make_profile(profile_list[RESNET], 29, resnet_gpu, resnet_cpu, resnet_cfg, resnet_data_G2C, resnet_data_C2G);
    make_profile(profile_list[RECURRENT], 6, rnn_gpu, rnn_cpu, rnn_cfg, rnn_data_G2C, rnn_data_C2G);

    return profile_list;
}

void check_registration(dnn_queue * dnn_list, int reg_fd, resource * gpu, resource *cpu, int baseline){
    reg_msg * msg = (reg_msg *)malloc(sizeof(reg_msg));
        
    while( read(reg_fd, msg, 7*sizeof(int)) > 0){
        if(msg -> regist == 1) regist(dnn_list, msg, baseline); 
        else de_regist(dnn_list, msg, gpu, cpu);
    }
}

void regist(dnn_queue * dnn_list, reg_msg * msg, int baseline){
    dnn_info * dnn = (dnn_info *)malloc(sizeof(dnn_info));

    dnn -> id = dnn_list -> count;

    dnn -> pid = msg -> pid;
    dnn -> layers = msg -> layers;
    dnn -> type = msg -> type;
    dnn -> period = msg -> period;
    dnn -> priority = msg -> priority;
    dnn -> current_layer = -1;
    dnn -> assigned = 1;
    dnn -> cut = msg->cut;
    
    if (dnn->cut != -2){
        dnn->default_cfg = (int *)malloc(sizeof(int)*dnn->layers);
        if (baseline == 4){
            memset(dnn->default_cfg, 0, sizeof(int)*dnn->layers);
            for(int i = 0; i < dnn->cut; i++) dnn->default_cfg[i] = 1;
        }
        if (baseline == 5){
            memset(dnn->default_cfg, 1, sizeof(int)*dnn->layers);
            for(int i  =0; i< dnn->cut; i++) dnn->default_cfg[i] = 0;
        }
    }
    
    
    debug_print("======== REGISTRATION ========\n");
    debug_print("[ID]     %3d\n", dnn-> id);
    debug_print("[PID]    %3d\n", dnn-> pid);
    debug_print("[Layers] %3d\n", dnn-> layers);
    debug_print("[Type]   %s\n", get_dnn_name(dnn->type));
    debug_print("[Period] %3d\n", dnn->period);
    debug_print("[Cut]    %3d\n", dnn->cut);
    char req_fd_name[30];
    char dec_fd_name[30];

    snprintf(req_fd_name, 30,"/tmp/request_%d",dnn->pid);
    snprintf(dec_fd_name, 30,"/tmp/decision_%d",dnn->pid);

    dnn -> request_fd = open_channel(req_fd_name, O_RDONLY);
    dnn -> decision_fd = open_channel(dec_fd_name, O_WRONLY);
    
    dnn -> next = NULL;

    enDNNQueue(dnn_list, dnn);
    
}

void bubbleSort(QNode * start) 
{ 
    int swapped; 
    QNode *ptr1; 
    QNode *lptr = NULL; 
  
    /* Checking for empty list */
    if (start == NULL) 
        return; 
  
    do
    { 
        swapped = 0; 
        ptr1 = start; 
  
        while (ptr1->next != lptr) 
        { 
            if (ptr1->priority > ptr1->next->priority) 
            {  
                swap(ptr1, ptr1->next); 
                swapped = 1; 
            } 
            ptr1 = ptr1->next; 
        } 
        lptr = ptr1; 
    } 
    while (swapped); 
} 
  
/* function to swap data of two nodes a and b*/
void swap(QNode *a, QNode *b) 
{ 
    int tmp_layer = a->layer;
    int tmp_id = a->id;
    int tmp_priority = a->priority;

    a->layer = b->layer;
    a->id = b->id;
    a->priority = b->priority;

    b->layer = tmp_layer;
    b->id = tmp_id;
    b->priority = tmp_priority;
} 

void re_assign_priority(dnn_queue * dnn_list, resource * gpu , resource * cpu){
    if(gpu->waiting->front != NULL){
        Queue * waiting  = gpu->waiting;
        for(QNode * tmp = waiting->front; tmp != NULL; tmp= tmp->next){
            dnn_info * target = find_dnn_by_id(dnn_list, tmp->id);
            tmp->priority = target->priority;
        }


        bubbleSort(gpu->waiting->front); 
    }
    if(cpu->waiting->front != NULL){
        Queue * waiting = cpu->waiting;
        for(QNode * tmp = waiting->front; tmp != NULL; tmp = tmp->next){
            dnn_info * target = find_dnn_by_id(dnn_list ,tmp->id);
            tmp->priority = target->priority;
        }

        bubbleSort(cpu->waiting->front);
    }
}


void de_regist(dnn_queue * dnn_list, reg_msg * msg, resource * gpu, resource *cpu){
    // find dnn info by pid;
    // delete dnn 
    // send signal to go 
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
    
    debug_print("[request_handler] : [ID] %d [layer] %d \n", node -> id, request_layer);

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
        if(node->cut == -2){
            if(profile->cfg[request_layer] == GPU) enQueue(gpu->waiting, request_layer, node ->  id, node -> priority);
            else enQueue(cpu->waiting, request_layer, node -> id, node -> priority );
        }
        else{
            if(node->default_cfg[request_layer] == GPU) enQueue(gpu->waiting, request_layer, node -> id, node -> priority );
            else enQueue(cpu->waiting, request_layer, node ->  id, node -> priority);
        }
      }
      else node->current_layer = -1;
}

void send_release_time(dnn_queue * dnn_list){
    struct timespec current_time;
    clock_gettime(CLOCK_MONOTONIC, &current_time);
    
    for(dnn_info * tmp = dnn_list-> head; tmp != NULL; tmp = tmp->next){
       if (write(tmp->decision_fd, &current_time, sizeof(struct timespec)) < 0)
            perror("release_time ");
    }

}

void decision_handler(int target_id, dnn_queue * dnn_list, int decision){
    
    cpu_set_t core;
    CPU_ZERO(&core);
    if(decision == GPU) CPU_SET(2, &core);
    if(decision == CPU) CPU_SET(4, &core);
    
    dnn_info * target = find_dnn_by_id(dnn_list, target_id);
    
    if(target->assigned != decision){
        sched_setaffinity(target->pid, sizeof(cpu_set_t), &core);
        target->assigned = decision;
    }

    if( write(target->decision_fd,&decision,sizeof(int)) < 0){
        perror("decision_handler");  
    }
}

void update_deadline(dnn_info * dnn, double current_time){
    // milli period to micro period
    double micro_period = dnn->period * 1000;
    dnn-> deadline = current_time + micro_period;
    debug_print("Deadline update : [dnn] %s ,[current] %f, [deadline] %f\n", get_dnn_name(dnn->type), current_time, dnn -> deadline);
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
    
    double slack, future_wait, blocked, data_trans, limits;
    dnn_info * node;
    int target_id = -1;
    int target_layer = -1;
    double smallest = DBL_MAX;
    
    int prefer, non_prefer; 

    //debug_print( "==============[MIGRATION]==============\n");
    for(QNode * tmp = q->front; tmp != NULL; tmp = tmp -> next){
        node = find_dnn_by_id(dnn_list, tmp -> id);
        slack = node->deadline - current_time - workload_left(profile_list[node->type],tmp -> layer, node->layers);
        
        prefer = (From -> res_id ==GPU) ? profile_list[node->type] -> gpu_exec[tmp->layer] : profile_list[node->type] -> cpu_exec[tmp->layer];
        non_prefer = (From -> res_id == GPU) ? profile_list[node->type] -> cpu_exec[tmp->layer] : profile_list[node->type] -> gpu_exec[tmp->layer];
        
        //debug_print( "[ID] : %d\n", tmp -> id);
        //debug_print( "[Slack] : %f\n",slack);
        //debug_print( "[Prefer] : %d\n",prefer);
        //debug_print( "[Non_prefer] : %d\n", non_prefer);
        
        if( slack > non_prefer - prefer ){ /* first condidtion */
            
            future_wait = waiting(q, dnn_list, profile_list, current_time, From, tmp->id);
            blocked = blocking(q, dnn_list, profile_list, From, tmp->id);
            data_trans = data_transfer(dnn_list , profile_list, From, tmp->id, tmp->layer);
            
            //debug_print( "[Futer_wait] : %f\n", future_wait);
            //debug_print( "[Blocked] : %f\n", blocked);
            //debug_print( "[data_trans] : %f\n", data_trans);

            if ( future_wait + prefer > blocked + data_trans + non_prefer ){
                
                limits = limit(q,dnn_list, profile_list, current_time, From, tmp->id);
                
                //debug_print("[Limits] : %f\n", limits);
                if( limits > non_prefer+ blocked+ data_trans ){
                    if( slack <= smallest ){
                        target_id = tmp -> id;
                        target_layer = tmp -> layer;
                        smallest = slack;
                    }
                    //debug_print( "[Smallest] : %f\n", smallest);
                }
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

        debug_print("Migration : [ID] %d [Layer] %d \n",target_id, target_layer);
    }

    return target_id; 
}

double waiting(Queue * q, dnn_queue * dnn_list, dnn_profile ** profile_list, double current_time, resource * From, int target_id){
    double waited = 0;

    dnn_info * current = find_dnn_by_id(dnn_list, From->id);
    dnn_info * target = find_dnn_by_id(dnn_list, target_id);
    double current_wait = (From->res_id == GPU) ? profile_list[current->type]->gpu_exec[From->layer] - (current_time - From->scheduled) : profile_list[current->type]->cpu_exec[From->layer] - (current_time - From->scheduled);

    waited += current_wait;

    if(target->priority > current->priority){
        for(int i = From->layer+1 ; i < current->layers ; i++){
            if(current->cut == -2){
                if (profile_list[current->type]->cfg[i] == From->res_id) {
                    waited += (From->res_id == GPU) ? profile_list[current->type]->gpu_exec[i] : profile_list[current->type]->cpu_exec[i];
                }else{
                    break;
                }
            }
            else{
                if(current->default_cfg[i] == From->res_id) waited += From->res_id == GPU ? profile_list[current->type]->gpu_exec[i] : profile_list[current->type]->cpu_exec[i];
                else break;
            }
        }
    }
    
    for(QNode *tmp = q->front; tmp->priority < target->priority; tmp = tmp->next){
        dnn_info * dnn = find_dnn_by_id(dnn_list,tmp->id);
        for(int i = tmp->layer; i < dnn->layers ; i ++){
            if(current->cut == -2){
                if(profile_list[dnn->type]->cfg[i] == From->res_id) {
                    waited += From->res_id == GPU ? profile_list[dnn->type]->gpu_exec[i] : profile_list[dnn->type]->cpu_exec[i];
                }else{
                    break;
                }
            }
            else{
                if(dnn->default_cfg[i] == From->res_id) waited += From->res_id == GPU ? profile_list[dnn->type]->gpu_exec[i] : profile_list[dnn->type]->cpu_exec[i];
                else break;
            }
        }
    }

    return waited;
}

double limit(Queue * q, dnn_queue * dnn_list, dnn_profile ** profile_list, double current_time, resource * From , int target_id){
    //debug_print("==================== Limit ====================\n ");
    
    double waited = 0;
    dnn_info * current = find_dnn_by_id(dnn_list, From->id);
    dnn_info * target = find_dnn_by_id(dnn_list, target_id);

    double current_wait = From->res_id == GPU ? profile_list[current->type]->gpu_exec[From->layer] - (current_time - From->scheduled) : profile_list[current->type] -> cpu_exec[From->layer] - (current_time - From->scheduled);

    waited += current_wait;
    
    int islimit = 1;
    
    //debug_print(" [current_DNN] :%s\n", get_dnn_name(current->type));
    //debug_print(" [Waited] :%f\n", waited);
    if(target->priority > current -> priority){
        for(int i = From -> layer + 1;  i < current->layers; i ++){
            //debug_print(" [Layer] : %d\n",i);
            //debug_print(" [Assigned] : %d\n", profile_list[current->type] ->cfg[i]);
            if(profile_list[current->type] -> cfg[i] == From->res_id){
                waited += From->res_id == GPU ? profile_list[current->type]->gpu_exec[i] : profile_list[current->type]->cpu_exec[i];
            }
            else{
                //debug_print(" Limit has been reached\n");
                islimit = 0;
                break;
            }
            //debug_print(" [Waited] :%f\n", waited);
        }
    }
    

    if(islimit){
        for(QNode * tmp = q->front; tmp->priority < target->priority ; tmp = tmp->next){
            dnn_info * dnn = find_dnn_by_id(dnn_list, tmp->id);
            for(int i  = tmp->layer; i < dnn->layers; i++){
                if(islimit){
                    if(profile_list[dnn->type] -> cfg[i] == From -> res_id){
                        waited += From->res_id == GPU ? profile_list[dnn->type]->gpu_exec[i] : profile_list[dnn->type]->cpu_exec[i];
                    }
                    else{
                        islimit = 0;
                        break;
                    }
                }
            }
        }
    }


    return waited;
}

/*
    1.find target's execution window (current time, target.deadline)
    2.find task's which next release(deadline) is between them
    3. for 2) they search all layers
    4.search remaining layers which is currently activated
*/
double blocking(Queue * q, dnn_queue * dnn_list,dnn_profile ** profile_list, resource * From, int target_id){
    dnn_info * target = find_dnn_by_id(dnn_list, target_id);

    double limit = target -> deadline;
    double biggest = 0;

    for(dnn_info * tmp = dnn_list->head; tmp != NULL; tmp = tmp ->next){
        if(tmp != target){
            if(tmp -> deadline < limit){
                dnn_profile * tmp_profile = profile_list[tmp->type];
                for(int i = 0; i < tmp->layers; i++){
                    double now = From->res_id == GPU ? tmp_profile -> gpu_exec[i] : tmp_profile -> cpu_exec[i] ;
                    if ( biggest < now) biggest = now;
                }        
            }else{
                if(tmp -> current_layer != -1){
                    dnn_profile * tmp_profile = profile_list[tmp->type];
                    for(int i = tmp->current_layer; i < tmp->layers; i++){
                        double now = From->res_id == GPU ? tmp_profile -> gpu_exec[i] : tmp_profile -> cpu_exec[i];
                        if (biggest < now) biggest = now;
                    }
                }
            }
        }
    }

    return biggest;
}

double data_transfer(dnn_queue * dnn_list, dnn_profile **profile_list, resource *From ,int target_id, int target_layer){
    int go, back;
    dnn_info * target = find_dnn_by_id(dnn_list, target_id);
    go = 0;
    
    if( target_layer != 0 && target->assigned == From->res_id) go = (From->res_id == GPU) ? profile_list[target->type]->G2C[target_layer-1] : profile_list[target->type]->C2G[target_layer-1];
    
    back = 0;
    
    for(int i = target_layer; i < target->layers -1; i++){
        if(From->res_id == GPU) {
            if(profile_list[target->type]->C2G[i] > back) back = profile_list[target->type]->C2G[i];
        }
        else{
            if(profile_list[target->type]->G2C[i] > back) back = profile_list[target->type]->G2C[i];
        }
    }
    return go + back;
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






