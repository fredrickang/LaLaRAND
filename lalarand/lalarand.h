#ifndef _LALARADN_H_
#define _LALARADN_H_

typedef enum{
    IDLE, BUSY
}STATE;

typedef enum {
    YOLOt, EXTRACTION, RESNET, RECURRENT 
}DNN_TYPE;

typedef struct _DNN_PROFILE{
    int * gpu_exec;
    int * cpu_exec;
    int *cfg;
}dnn_profile;

typedef struct _DNN_INFO{
    int pid;
    int id;
    int layers;
    DNN_TYPE type; 
    int period;
    double deadline;
    int request_fd;
    int decision_fd;
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
    struct QNode * next;
    struct QNode * prev;
}QNode;

typedef struct Queue{
    int count;
    QNode * front, *rear;
}Queue;

typedef struct _MSG_PACKET{
    int regist;
    int pid;
    int layers;
    DNN_TYPE type;
}reg_msg;

typedef struct _RESOURCE{
    STATE state;
    Queue * waiting;
    int id;
}resource;

#endif
