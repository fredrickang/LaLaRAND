#ifndef _LALARADN_H_
#define _LALARADN_H_

typedef enum{
    IDLE, BUSY, ALIVE, TERM
}STATE;

typedef enum {
    YOLOt, EXTRACTION, RESNET, RECURRENT, ALEXNET, DARKNET, LENET
}DNN_TYPE;

typedef struct _DNN_PROFILE{
    int * gpu_exec;
    int * cpu_exec;
    int *cfg;
    int * G2C;
    int * C2G;
}dnn_profile;

typedef struct _DNN_INFO{
    int pid;
    int id;
    STATE state;
    int layers;
    DNN_TYPE type; 
    int period;
    double deadline;
    int request_fd;
    int decision_fd;
    int priority;
    int current_layer;
    int assigned;
    int * default_cfg;
    struct _DNN_INFO * next;
}dnn_info;

typedef struct _DNN_QUEUE{
    int count;
    dnn_info * head;
}dnn_queue;

// Queue Definition
typedef struct QNode{
    int layer;
    int id;
    int priority;
    struct QNode * next;
}QNode;

typedef struct Queue{
    int count;
    QNode * front;
}Queue;

typedef struct _MSG_PACKET{
    int regist;
    int pid;
    int layers;
    DNN_TYPE type;
    int period;
    int priority;
    int cut;
}reg_msg;

typedef struct _REQUEST_MSG{
    int request_layer;
    int request_type;
}req_msg;

typedef struct _RESOURCE{
    int res_id;
    STATE state;
    Queue * waiting;
    int id;
    int layer;
    double scheduled;
}resource;

#endif
