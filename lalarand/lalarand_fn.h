#ifndef _LALALRAND_FN_H_
#define _LALARAND_FN_H_
#include "lalarand.h"

///// dnn queue ////
dnn_queue * createDNNQueue();
void enDNNQueue(dnn_queue * dnn_list, dnn_info* dnn);
void deleteDNN(dnn_queue * dnn_list, dnn_info* dnn);

///// resource /////
resource * createResource();

//// waiting queue ////
QNode* newNode (int layer, int id);
Queue * createQueue();
void enQueue(Queue *q, int layer, int id);
void deleteNode(Queue *q , QNode * del);
int deQueue(Queue * q, dnn_queue * dnn_list, dnn_profile ** profile_list, double current_time, resource * res);
dnn_info * find_dnn_by_id(dnn_queue * dnn_list, int id);
void print_queue(char * name, Queue * q);

///// Utils ////
void del_arg(int argc, char **argv, int index);
int find_int_arg(int argc, char **argv, char *arg, int def);
double get_time_point();

//// LaLaRAND ////
void make_profile(dnn_profile * tmp, int layers, int *gpu, int *cpu, int *cfg);
dnn_profile ** make_profile_list();
void check_registration(dnn_queue * dnn_list, int reg_fd);
void regist(dnn_queue * dnn_list, reg_msg * msg);
void de_regist(dnn_queue * dnn_list, reg_msg *msg);
int check_request(dnn_queue * dnn_list, fd_set *readfds);
int migration(Queue * q, dnn_queue * dnn_list, dnn_profile** profile_list, double current_time, resource * res);
void request_handler(dnn_info * node, resource * gpu, resource * cpu, dnn_profile * profile, double current_time);
void decision_handler(int target_id, dnn_queue * dnn_list, int decision);
void update_deadline(dnn_info * dnn, double current_time);
void update_deadline_all(dnn_queue * dnn_list, double current_time);
char* get_dnn_name(DNN_TYPE type);
double workload_left(dnn_profile * profile, int current_layer, int layer_num);

int open_channel(char * pipe_name,int mode);
void close_channel(char * pipe_name);
void close_channels(dnn_info ** dnn_list, int dnns);
#endif
