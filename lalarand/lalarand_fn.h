#ifndef _LALALRAND_FN_H_
#define _LALARAND_FN_H_
#include "lalarand.h"


void logging(int baseline, int algo, int index);
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
int deQueue_hiding(Queue *q, double current_time, resource *res, resource * mem);
int deQueue_algo(Queue * q, dnn_queue * dnn_list, dnn_profile ** profile_list, double current_time, resource * res);
int deQueue(Queue * q, double current_time, resource * res);
dnn_info * find_dnn_by_id(dnn_queue * dnn_list, int id);
dnn_info * find_dnn_by_pid(dnn_queue * dnn_list, int pid);
int find_node_by_id(Queue *q, int id);
void print_queue(char * name, Queue * q);
void print_list(char * name, dnn_queue * dnn_list);
void re_assign_priority(dnn_queue * dnn_list, resource * gpu, resource * cpu);

///// Utils ////
void del_arg(int argc, char **argv, int index);
int find_int_arg(int argc, char **argv, char *arg, int def);
double get_time_point();
void bubbleSort(QNode * start);
void swap(QNode * a, QNode *b);

void send_release_time(dnn_queue * dnn_list);
//// LaLaRAND ////
void make_profile(dnn_profile * tmp, int layers, int *gpu, int *cpu, int *cfg);
dnn_profile ** make_profile_list(int mode);
void check_registration(dnn_queue * dnn_list, int reg_fd, resource * gpu, resource *cpu,  int baseline);
void regist(dnn_queue * dnn_list, reg_msg * msg,int baseline);
void de_regist(dnn_queue * dnn_list, reg_msg *msg, resource * gpu, resource * cpu);
int check_request(dnn_queue * dnn_list, fd_set *readfds, int sync);
int migration(Queue * q, dnn_queue * dnn_list, dnn_profile** profile_list, double current_time, resource * From, resource * To);
void request_handler(int hiding, dnn_info * node, resource * gpu, resource * cpu, resource * mem, dnn_profile * profile, double current_time);
void decision_handler(int target_id, dnn_queue * dnn_list, int decision);
void update_deadline(dnn_info * dnn, double current_time);
void update_deadline_all(dnn_queue * dnn_list, double current_time);
char* get_dnn_name(DNN_TYPE type);
double workload_left(dnn_profile * profile, int current_layer, int layer_num);
int make_fdset(fd_set *readfds,int reg_fd, dnn_queue * dnn_list);
double waiting(Queue * q, dnn_queue * dnn_list, dnn_profile ** profile_list, double current_time, resource * res, int target_id);
double limit(Queue * q, dnn_queue * dnn_list, dnn_profile ** profile_list, double current_time, resource * res, int target_id);
double blocking(Queue * q, dnn_queue * dnn_list,dnn_profile ** profile_list, resource * From, int target_id);
int open_channel(char * pipe_name,int mode);
void close_channel(char * pipe_name);
void close_channels(dnn_info * dnn);
double data_transfer(dnn_queue * dnn_list ,dnn_profile ** profile_list, resource *From, int target_id, int target_layer);


#endif
