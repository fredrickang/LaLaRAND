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
#define GPU 1
#define CPU 0

typedef enum{
    IDLE, BUSY
}STATE;

typedef struct _RESOURCE{
    STATE state;
    int id;
}resource;

typedef struct _DNN_PROFILE{
    int * gpu_exec;
    int * cpu_exec;
    int *cfg;
}dnn_profile;

typedef enum {
    YOLOt, EXTRACTION, RESNET, RECURRENT 
}DNN_TYPE;

// Queue Definition
typedef struct QNode{
    dnn_info *dnn;
    int layer;
    int id;
    struct QNode * next;
}QNode;

typedef struct Queue {
    int count;
    QNode * front, *rear;
}Queue;

typedef struct _DNN_INFO{
    int pid;
    int layers;
    DNN_TYPE type; 
    dnn_profile *profile; 
    int period;
    clock_t deadline;
}dnn_info;

typedef struct _MSG_PACKET{
    int pid;
    int layers;
    DNN_TYPE type;
}msg;

QNode*  newNode(int pid, int layer, int id);
Queue*  createQueue(char *name);
void    enQueue(Queue *q, int pid, int layer, int id);
QNode*  deQueue(Queue * q);

void    del_arg(int argc, char **argv, int index);
int     find_int_arg(int argc, char **argv, char *arg, int def);
char*   get_dnn_name(DNN_TYPE type);

dnn_info** network_register(int register_fd, int dnns);
void    adding_profile(dnn_info** dnn_list, int dnns);
int     open_channel(char * pipe_name,int mode);
void    close_channel(char * pipe_name);

void    update_deadliine(dnn_info * dnn, current_time);

int main(int argc, char **argv){
    
    //Input argument handling
    int dnns = find_int_arg(argc, argv, "-dnns", 1);
    int Sync = find_int_arg(argc, argv, "-sync", 1);
    
    int register_fd;
    dnn_info ** dnn_list;
    
    //CPU affinity to core 1
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(1, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);

    //Registeration part
    register_fd = open_channel("./lalarand_register",O_RDWR);
    dnn_list = network_register(register_fd, dnns);
    close_channel("./lalarand_register");
    
    //Adding additional information
    adding_profile(dnn_list, dnns);
    
    //Open Request channel
    int request_fd[dnns];
    char request[30];
    for(int i =0; i < dnns; i++){
        snprintf(request, 30, "lalarand_request_%d", dnn_list[i]->pid);
        request_fd[i] = open_channel(request, O_RDONLY); 
    }
    puts("\nrequest channel has been openned\n");
    
    //Open Decision channel
    int decision_fd[dnns];
    char decision[30];
    for(int i =0; i < dnns; i++){
        snprintf(decision, 30, "lalarand_decision_%d", dnn_list[i]->pid);
        decision_fd[i] = open_channel(decision, O_RDWR);
    }
    puts("\ndecision channel has been openned\n"); 
    

    int request_layer = -1;
    int resource = 1;
    int err;
   
    Queue * gpu_request = createQueue("GPU_REQUEST");
    Queue * cpu_request = createQueue("CPU_REQEUST");
    resource * gpu = (resource *)malloc(sizeof(resource));
    resource * cpu = (resource *)malloc(sizeof(resource)); 
    
    gpu -> state = IDLE;
    cpu -> state = IDLE;

    gpu -> id = -1;
    cpu -> id = -1;
    
    QNode * target;
    
    
    fd_set readfds, writefds;
    int state;
    clock_t current_time = 0;
    while(1){
        
        current_time = clock();

        FD_ZERO(&readfds); 
        for(int i = 0 ; i < dnns; i++)
            FD_SET(request_fd[i], &readfds);
       
        state = select(request_fd[dnns-1] + 1, &readfds, NULL, NULL, 0);
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
                        
                        FD_CLR(request_fd[i],&readfds);
                        
                        if( read(request_fd[i], &request_layer, sizeof(int)) < 0){
                            perror("read error : ");    
                            exit(-1);
                        }
                        
                        if(request_layer == 0) update_deadline(dnn_list[i], current_time);
                        
                        if( gpu -> state == BUSY && gpu -> id == i){
                            gpu -> state = IDLE;
                            gpu -> id = -1;
                        }
                        
                        printf("Frome %d pid, layer %d has been requested\n", dnn_list[i]->pid, request_layer);
                        
                        if(dnn_list[i]->profile->cfg[request_layer] == GPU) enQueue(gpu_request, dnn_list[i]->pid, request_layer, i);
                        else enQueue(cpu_request, dnn_list[i]->pid, request_layer, i); 
                    
                    }
                }
        }
        
        if(!(Sync && (gpu_request->count + cpu_request->count) < dnns)){
            target = NULL;
            
            if(Sync) for(int i =0 ; i < dnns ; i++) update_deadline(dnn_list[i], current_time);
            
            if(gpu -> state == IDLE) target = deQueue(gpu_request, current_time, gpu);    
            
            if(target){
                if(write(decision_fd[target->id], &resource, sizeof(int)) < 0){
                    perror("write error : ");
                    exit(-1);
                }
                printf("Decision has been sent to %d as %d\n", target->pid, resource);
            } 

            Sync = 0;
        }

    }
    

    for(int i = 0; i < dnns; i++){
        snprintf(request, 30, "lalarand_request_%d", dnn_list[i]->pid);
        snprintf(decision, 30, "lalarand_decision_%d", dnn_list[i]->pid);
        close_channel(request);
        close_channel(decision);
    }
}

QNode* newNode(dnn_info *info, int layer, int id){
    QNode * tmp = (QNode *)malloc(sizeof(QNode));
    tmp -> dnn = info;
    tmp -> layer = layer;
    tmp -> id = id;
    tmp -> next = NULL;

    return tmp;

}

Queue * createQueue(char *name){

    Queue * q = (Queue *)malloc(sizeof(Queue));
    q->front = q->rear =  NULL;
    q-> count = 0;
    printf("%s queue has been made\n", name);
    return q;
    
}

void enQueue(Queue *q, dnn_info * info, int layer, int id){
    QNode * tmp = newNode(info, layer, id);
    q->count ++;    
    if(q->rear == NULL){
        q->front = q->rear = tmp;
        return;
    }
    
    q->rear->next = tmp;
    q->rear = tmp;
}

QNode* deQueue(Queue * q){
    if(q->front == NULL)
        return NULL;

    QNode * tmp = q->front;

    q->front = q->front->next;

    if(q->front == NULL)
        q->rear = NULL;
    q->count --;
    return tmp;
}

// .Queue Definition


// Argument handling
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

// .Arguement handling



dnn_info ** network_register(int register_fd, int dnns){
    
    int count =0;
    
    puts("");
    printf("Waiting for %d dnns to be registered\n",dnns);
    
    dnn_info ** dnn_list = (dnn_info **)malloc(sizeof(dnn_info *)* dnns);
    msg * dummy = (msg *)malloc(sizeof(msg));

    for(int i =0 ; i < dnns ; i ++)
        dnn_list[i] = (dnn_info *)malloc(sizeof(dnn_info));
   
    while(count < dnns){
       
        if(read(register_fd, dummy, sizeof(dummy)) < 0){
           perror(" Registeration : ");
           exit(-1);
       }
       
       dnn_list[count]->pid = dummy->pid;
       dnn_list[count]->layers = dummy->layers;
       dnn_list[count]->type = dummy->type;
       dnn_list[count]->period = 220;
       puts("======================");
       printf("%d/%d\n",count+1, dnns);
       printf("[DNN] : %s \n", get_dnn_name(dnn_list[count]->type));
       printf("[PID] : %d \n", dnn_list[count]->pid);
       printf("[Layer] : %d \n",dnn_list[count]->layers);
       printf("[Period] : %d \n", dnn_list[count]->period);
       puts("\nregistered\n"); 
       count ++;
    }
    
    for(int i =0; i <dnns ; i++) kill(dnn_list[i]->pid, SIGCONT);
    
    return dnn_list;
}

int open_channel(char * pipe_name,int mode){
    int pipe_fd;
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

void make_profile(dnn_profile * tmp, int layers, int *gpu, int *cpu, int *cfg){
    
    tmp->gpu_exec = (int *)malloc(sizeof(int) * layers);
    tmp->cpu_exec = (int *)malloc(sizeof(int) * layers);
    tmp->cfg = (int *)malloc(sizeof(int) * layers);

    memcpy(tmp->gpu_exec, gpu, sizeof(int) *layers);
    memcpy(tmp->cpu_exec, cpu, sizeof(int) *layers);
    memcpy(tmp->cfg , cfg, sizeof(int) * layers);
}

void adding_profile(dnn_info** dnn_list, int dnns){
    
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
    
    for(int i = 0 ; i < dnns ; i++){
        dnn_list[i]->profile = (dnn_profile *)malloc(sizeof(dnn_profile));
        switch(dnn_list[i]->type){
            case YOLOt:
                make_profile(dnn_list[i]->profile, 24, yolo_gpu, yolo_cpu, yolo_cfg);
                break;
            case EXTRACTION:
                make_profile(dnn_list[i]->profile, 28, extraction_gpu, extraction_cpu, extraction_cfg);
                break;
            case RESNET:
                make_profile(dnn_list[i]->profile, 29, resnet_gpu, resnet_cpu, resnet_cfg);
                break;
            case RECURRENT:
                make_profile(dnn_list[i]->profile, 5, rnn_gpu, rnn_cpu, rnn_cfg);
                break;
        }
    }
    
}

void update_deadline(dnn_info * dnn, current_time){
    dnn -> deadline = current_time + dnn -> period;
}

char* get_dnn_name(DNN_TYPE type){
    if(type == 0) return "YOLO";
    if(type == 1) return "EXTRACTION";
    if(type == 2) return "RESNET";
    if(type == 3) return "RNN";
}
