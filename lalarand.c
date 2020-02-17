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
typedef struct QNode{
    int pid;
    int layer;
    int id;
    struct QNode * next;
}QNode;

typedef struct Queue {
    int count;
    QNode * front, *rear;
}Queue;

QNode* newNode(int pid, int layer, int id){
    QNode * tmp = (QNode *)malloc(sizeof(QNode));
    tmp -> pid = pid;
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

void enQueue(Queue *q, int pid, int layer, int id){
    QNode * tmp = newNode(pid, layer, id);
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


typedef struct _Dnn_info{
    int pid;
    int layer_num;    
}dnn_info;


// ********Utils ************* 
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
// ******* 





dnn_info ** network_register(int register_fd, int dnns){
    
    int count =0;
    // make storage
    
    puts("");
    printf("Waiting for %d dnns to be registered\n",dnns);
    
    dnn_info ** dnn_list = (dnn_info **)malloc(sizeof(dnn_info *)* dnns);
    for(int i =0 ; i < dnns ; i ++)
        dnn_list[i] = (dnn_info *)malloc(sizeof(dnn_info));
    
    //read
    while(count < dnns){
       if(read(register_fd, dnn_list[count], sizeof(dnn_list[count])) < 0){
           puts("[ERROR]Fail to read register info");
           exit(-1);
       }
       //log
       puts("======================");
       printf("%d/%d",count+1, dnns);
       printf("[pid] %d, [layers] %d\n", dnn_list[count]->pid, dnn_list[count]->layer_num);
       puts("registered\n"); 
       count ++;
    }
    
    for(int i =0; i <dnns ; i++){
        kill(dnn_list[i]->pid, SIGCONT);
    }
    return dnn_list;
}

// mode : 0 write , 1 read
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

int main(int argc, char **argv){
    int dnns = find_int_arg(argc, argv, "-dnns", 1);
    int Sync = find_int_arg(argc, argv, "-sync", 1);
    
    int register_fd;
    dnn_info ** dnn_list;
    
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(1, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);


    // registeration
    register_fd = open_channel("./lalarand_register",O_RDWR);
    dnn_list = network_register(register_fd, dnns);
    close_channel("./lalarand_register");
    
    // request channel
    int request_fd[dnns];
    char request[30];
    for(int i =0; i < dnns; i++){
        snprintf(request, 30, "lalarand_request_%d", dnn_list[i]->pid);
        request_fd[i] = open_channel(request, O_RDONLY); 
    }
    puts("\nrequest channel has been openned\n");
    
    // decision channel
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
    Queue * request_q = createQueue("request");
    QNode * target;
    
    
    fd_set readfds, writefds;
    int state;

    while(1){
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
                        if( read(request_fd[i], &request_layer, sizeof(int)) < 0)
                            perror("read error : ");
                        else{
                            printf("Frome %d pid, layer %d has been requested\n", dnn_list[i]->pid, request_layer);
                            enQueue(request_q, dnn_list[i]->pid, request_layer, i);
                        }
                    }
                }
        }
        
        if(Sync && request_q -> count < dnns){
        }
        else{
            Sync = 0;
            target = deQueue(request_q);
            
            if(target){
                if(write(decision_fd[target->id], &resource, sizeof(int)) < 0)
                    perror("write error : ");
                else{
                    printf("Decision has been sent to %d as %d\n", target->pid, resource);
                }
            } 
        }

    }
    

    for(int i = 0; i < dnns; i++){
        snprintf(request, 30, "lalarand_request_%d", dnn_list[i]->pid);
        snprintf(decision, 30, "lalarand_decision_%d", dnn_list[i]->pid);
        close_channel(request);
        close_channel(decision);
    }
}

