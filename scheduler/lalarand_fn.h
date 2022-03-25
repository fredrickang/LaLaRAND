#ifndef _LALALRAND_FN_H_
#define _LALARAND_FN_H_
#include "lalarand.h"

void set_priority(int priority);
void set_affinity(int core);

///// dnn queue ////
dnn_queue * createDNNQueue();
void enDNNQueue(dnn_queue * dnn_list, dnn_info* dnn);
void deleteDNN(dnn_queue * dnn_list, dnn_info* dnn);
void setDNNpriority(dnn_queue * dnn_list);

///// resource /////
resource * createResource(int res_id);

//// waiting queue ////
QNode* newNode (int layer, int id, int priority);
Queue * createQueue();
void enQueue(Queue *q, int layer, int id, int priority);
int deQueue(Queue * q, double current_time, resource * res);
dnn_info * find_dnn_by_id(dnn_queue * dnn_list, int id);
dnn_info * find_dnn_by_pid(dnn_queue * dnn_list, int pid);
int find_node_by_id(Queue *q, int id);
void print_queue(char * name, Queue * q);
void print_list(char * name, dnn_queue * dnn_list);
void del_arg(int argc, char **argv, int index);
int find_int_arg(int argc, char **argv, char *arg, int def);
double get_time_point();
void send_release_time(dnn_queue * dnn_list);
void check_registration(dnn_queue * dnn_list, int reg_fd, resource * gpu, resource *cpu);
void regist(dnn_queue * dnn_list, reg_msg * msg);
void de_regist(dnn_queue * dnn_list, reg_msg *msg, resource * gpu, resource * cpu);
void request_handler(dnn_info * node, resource * gpu, resource * cpu, double current_time);
void decision_handler(int target_id, dnn_queue * dnn_list, int decision);
void update_deadline(dnn_info * dnn, double current_time);
void update_deadline_all(dnn_queue * dnn_list, double current_time);
char* get_resource_name(int id);
int make_fdset(fd_set *readfds,int reg_fd, dnn_queue * dnn_list);
int open_channel(char * pipe_name,int mode);
void close_channel(char * pipe_name);
void close_channels(dnn_info * dnn);
void read_default_cfg(int pid, int * default_cfg);
#endif
