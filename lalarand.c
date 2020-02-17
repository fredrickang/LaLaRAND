#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>


typedef struct QNode{
    int pid;
    int layer;
    int id;
    struct QNode * next;
}QNode;

typedef struct Queue {
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

Queue * createQueue(){
    Queue * q = (Queue *)malloc(sizeof(Queue));
    q->front = q->rear =  NULL;
    return q;
}

void enQueue(Queue *q, int pid, int layer, int id){
    QNode * tmp = newNode(pid, layer, id);

    if(q->rear = NULL){
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

    if(q->front = NULL)
        q->rear = NULL;

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

    // registeration
    register_fd = open_channel("./lalarand_register",O_RDWR);
    dnn_list = network_register(register_fd, dnns);
    close_channel("./lalarand_register");
    
    // request channel
    int request_fd[dnns];
    char request[30];
    for(int i =0; i < dnns; i++){
        snprintf(request, 30, "lalarand_request_%d", dnn_list[i]->pid);
        request_fd[i] = open_channel(request,O_RDONLY | O_NONBLOCK); 
    }
    puts("\nrequest channel has been openned\n");
    
    // decision channel
    int decision_fd[dnns];
    char decision[30];
    for(int i =0; i < dnns; i++){
        snprintf(decision, 30, "lalarand_decision_%d", dnn_list[i]->pid);
        decision_fd[i] = open_channel(decision,O_WRONLY);
    }
    puts("\ndecision channel has been openned\n"); 
    
    int request_layer;
    int rev;
    int resource = 1;
    Queue * request_q = createQueue();
    QNode * target;
    while(1){

        // Collecting request
        if(Sync){
            /* First job , First layer request */
            for(int i =0 ; i < dnns ; i++){
                while(read(request_fd[i], &request_layer, sizeof(int)) == -1);
                enQueue(request_q, dnn_list[i]->pid, request_layer, i);
            }
            Sync = 0;
            printf("Synchronized release ready\n");
        }
        else{
            for(int i = 0 ; i < dnns; i++){
                if(read(request_fd[i], &request_layer, sizeof(int)) != -1)
                    enQueue(request_q, dnn_list[i]->pid, request_layer, i);
            }
        }
        
        // Make decision
        target = deQueue(request_q);
        
        // Send decision
        if(target){
            write(decision_fd[target->id], &resource, sizeof(int));
        }
    

    }
    

    for(int i = 0; i < dnns; i++){
        snprintf(request, 30, "lalarand_request_%d", dnn_list[i]->pid);
        snprintf(decision, 30, "lalarand_decision_%d", dnn_list[i]->pid);
        close_channel(request);
        close_channel(decision);
    }
}

