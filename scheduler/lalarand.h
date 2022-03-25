#ifndef _LALARADN_H_
#define _LALARADN_H_

typedef enum{
    IDLE, BUSY
}STATE;

typedef struct _DNN_INFO{
    int pid;
    int id;
    int layers; 
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
    QNode * rear;
}Queue;

typedef struct _MSG_PACKET{
    int regist;
    int pid;
    int layers;
    int period;
    int priority;
}reg_msg;

typedef struct _REQUEST_MSG{
    int request_layer;
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
