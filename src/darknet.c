#define _GNU_SOURCE
#include "darknet.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>
#include <sys/mman.h>
#include <pthread.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <sys/resource.h>
#include <sched.h>

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>

#endif

#include "parser.h"
#include "utils.h"
#include "dark_cuda.h"
#include "blas.h"
#include "connected_layer.h"

#define MYMUTEX "/mymutex"
#define MYQUEUE  "/mycond"

extern void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top);
extern void run_voxel(int argc, char **argv);
extern void run_yolo(int argc, char **argv);
extern void run_detector(int argc, char **argv);
extern void run_coco(int argc, char **argv);
extern void run_writing(int argc, char **argv);
extern void run_captcha(int argc, char **argv);
extern void run_nightmare(int argc, char **argv);
extern void run_dice(int argc, char **argv);
extern void run_compare(int argc, char **argv);
extern void run_classifier(int argc, char **argv);
extern void run_char_rnn(int argc, char **argv);
extern void run_vid_rnn(int argc, char **argv);
extern void run_tag(int argc, char **argv);
extern void run_cifar(int argc, char **argv);
extern void run_go(int argc, char **argv);
extern void run_art(int argc, char **argv);
extern void run_super(int argc, char **argv);

//global variables for runtime switching.
extern int test_extern = 1972;
extern int *test_extern_arr = NULL;
extern int * queue = NULL;
extern pthread_mutex_t *gpu_lock = NULL;
extern int N = 0;

// processes identifier & shared memory
extern int identifier = -1;
extern int **shmem_rescfg = NULL;
extern char **shmem_mlist = NULL;
extern int *shmem_pid = NULL;
extern struct timespec *shmem_timer = NULL;

//DetectorParameter structure for multi-threading.
DetectorParams *_g_detector_params;
//!end of DetectorParameter structure init.

void thread_test_function(int a, int b)
{
    printf("this is a : %d\n, this is b : %d\n");
}

void average(int argc, char *argv[])
{
    char *cfgfile = argv[2];
    char *outfile = argv[3];
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    network sum = parse_network_cfg(cfgfile);

    char *weightfile = argv[4];
    load_weights(&sum, weightfile);

    int i, j;
    int n = argc - 5;
    for(i = 0; i < n; ++i){
        weightfile = argv[i+5];
        load_weights(&net, weightfile);
        for(j = 0; j < net.n; ++j){
            layer l = net.layers[j];
            layer out = sum.layers[j];
            if(l.type == CONVOLUTIONAL){
                int num = l.n*l.c*l.size*l.size;
                axpy_cpu(l.n, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(num, 1, l.weights, 1, out.weights, 1);
                if(l.batch_normalize){
                    axpy_cpu(l.n, 1, l.scales, 1, out.scales, 1);
                    axpy_cpu(l.n, 1, l.rolling_mean, 1, out.rolling_mean, 1);
                    axpy_cpu(l.n, 1, l.rolling_variance, 1, out.rolling_variance, 1);
                }
            }
            if(l.type == CONNECTED){
                axpy_cpu(l.outputs, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, out.weights, 1);
            }
        }
    }
    n = n+1;
    for(j = 0; j < net.n; ++j){
        layer l = sum.layers[j];
        if(l.type == CONVOLUTIONAL){
            int num = l.n*l.c*l.size*l.size;
            scal_cpu(l.n, 1./n, l.biases, 1);
            scal_cpu(num, 1./n, l.weights, 1);
                if(l.batch_normalize){
                    scal_cpu(l.n, 1./n, l.scales, 1);
                    scal_cpu(l.n, 1./n, l.rolling_mean, 1);
                    scal_cpu(l.n, 1./n, l.rolling_variance, 1);
                }
        }
        if(l.type == CONNECTED){
            scal_cpu(l.outputs, 1./n, l.biases, 1);
            scal_cpu(l.outputs*l.inputs, 1./n, l.weights, 1);
        }
    }
    save_weights(sum, outfile);
}

void speed(char *cfgfile, int tics)
{
    if (tics == 0) tics = 1000;
    network net = parse_network_cfg(cfgfile);
    set_batch_network(&net, 1);
    int i;
    time_t start = time(0);
    image im = make_image(net.w, net.h, net.c);
    for(i = 0; i < tics; ++i){
        network_predict(net, im.data);
    }
    double t = difftime(time(0), start);
    printf("\n%d evals, %f Seconds\n", tics, t);
    printf("Speed: %f sec/eval\n", t/tics);
    printf("Speed: %f Hz\n", tics/t);
}

void operations(char *cfgfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    int i;
    long ops = 0;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            ops += 2l * l.n * l.size*l.size*l.c * l.out_h*l.out_w;
        } else if(l.type == CONNECTED){
            ops += 2l * l.inputs * l.outputs;
		} else if (l.type == RNN){
            ops += 2l * l.input_layer->inputs * l.input_layer->outputs;
            ops += 2l * l.self_layer->inputs * l.self_layer->outputs;
            ops += 2l * l.output_layer->inputs * l.output_layer->outputs;
        } else if (l.type == GRU){
            ops += 2l * l.uz->inputs * l.uz->outputs;
            ops += 2l * l.uh->inputs * l.uh->outputs;
            ops += 2l * l.ur->inputs * l.ur->outputs;
            ops += 2l * l.wz->inputs * l.wz->outputs;
            ops += 2l * l.wh->inputs * l.wh->outputs;
            ops += 2l * l.wr->inputs * l.wr->outputs;
        } else if (l.type == LSTM){
            ops += 2l * l.uf->inputs * l.uf->outputs;
            ops += 2l * l.ui->inputs * l.ui->outputs;
            ops += 2l * l.ug->inputs * l.ug->outputs;
            ops += 2l * l.uo->inputs * l.uo->outputs;
            ops += 2l * l.wf->inputs * l.wf->outputs;
            ops += 2l * l.wi->inputs * l.wi->outputs;
            ops += 2l * l.wg->inputs * l.wg->outputs;
            ops += 2l * l.wo->inputs * l.wo->outputs;
        }
    }
    printf("Floating Point Operations: %ld\n", ops);
    printf("Floating Point Operations: %.2f Bn\n", (float)ops/1000000000.);
}

void oneoff(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    int oldn = net.layers[net.n - 2].n;
    int c = net.layers[net.n - 2].c;
    net.layers[net.n - 2].n = 9372;
    net.layers[net.n - 2].biases += 5;
    net.layers[net.n - 2].weights += 5*c;
    if(weightfile){
        load_weights(&net, weightfile);
    }
    net.layers[net.n - 2].biases -= 5;
    net.layers[net.n - 2].weights -= 5*c;
    net.layers[net.n - 2].n = oldn;
    printf("%d\n", oldn);
    layer l = net.layers[net.n - 2];
    copy_cpu(l.n/3, l.biases, 1, l.biases +   l.n/3, 1);
    copy_cpu(l.n/3, l.biases, 1, l.biases + 2*l.n/3, 1);
    copy_cpu(l.n/3*l.c, l.weights, 1, l.weights +   l.n/3*l.c, 1);
    copy_cpu(l.n/3*l.c, l.weights, 1, l.weights + 2*l.n/3*l.c, 1);
    *net.seen = 0;
    save_weights(net, outfile);
}

void partial(char *cfgfile, char *weightfile, char *outfile, int max)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights_upto(&net, weightfile, max);
    }
    *net.seen = 0;
    save_weights_upto(net, outfile, max);
}

#include "convolutional_layer.h"
void rescale_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            rescale_weights(l, 2, -.5);
            break;
        }
    }
    save_weights(net, outfile);
}

void rgbgr_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            rgbgr_weights(l);
            break;
        }
    }
    save_weights(net, outfile);
}

void reset_normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL && l.batch_normalize) {
            denormalize_convolutional_layer(l);
        }
        if (l.type == CONNECTED && l.batch_normalize) {
            denormalize_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
        }
        if (l.type == LSTM && l.batch_normalize) {
            denormalize_connected_layer(*l.wf);
            denormalize_connected_layer(*l.wi);
            denormalize_connected_layer(*l.wg);
            denormalize_connected_layer(*l.wo);
            denormalize_connected_layer(*l.uf);
            denormalize_connected_layer(*l.ui);
            denormalize_connected_layer(*l.ug);
            denormalize_connected_layer(*l.uo);
		}
    }
    save_weights(net, outfile);
}

layer normalize_layer(layer l, int n)
{
    int j;
    l.batch_normalize=1;
    l.scales = (float*)calloc(n, sizeof(float));
    for(j = 0; j < n; ++j){
        l.scales[j] = 1;
    }
    l.rolling_mean = (float*)calloc(n, sizeof(float));
    l.rolling_variance = (float*)calloc(n, sizeof(float));
    return l;
}

void normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL && !l.batch_normalize){
            net.layers[i] = normalize_layer(l, l.n);
        }
        if (l.type == CONNECTED && !l.batch_normalize) {
            net.layers[i] = normalize_layer(l, l.outputs);
        }
        if (l.type == GRU && l.batch_normalize) {
            *l.input_z_layer = normalize_layer(*l.input_z_layer, l.input_z_layer->outputs);
            *l.input_r_layer = normalize_layer(*l.input_r_layer, l.input_r_layer->outputs);
            *l.input_h_layer = normalize_layer(*l.input_h_layer, l.input_h_layer->outputs);
            *l.state_z_layer = normalize_layer(*l.state_z_layer, l.state_z_layer->outputs);
            *l.state_r_layer = normalize_layer(*l.state_r_layer, l.state_r_layer->outputs);
            *l.state_h_layer = normalize_layer(*l.state_h_layer, l.state_h_layer->outputs);
            net.layers[i].batch_normalize=1;
        }
        if (l.type == LSTM && l.batch_normalize) {
            *l.wf = normalize_layer(*l.wf, l.wf->outputs);
            *l.wi = normalize_layer(*l.wi, l.wi->outputs);
            *l.wg = normalize_layer(*l.wg, l.wg->outputs);
            *l.wo = normalize_layer(*l.wo, l.wo->outputs);
            *l.uf = normalize_layer(*l.uf, l.uf->outputs);
            *l.ui = normalize_layer(*l.ui, l.ui->outputs);
            *l.ug = normalize_layer(*l.ug, l.ug->outputs);
            *l.uo = normalize_layer(*l.uo, l.uo->outputs);
            net.layers[i].batch_normalize=1;
        }
    }
    save_weights(net, outfile);
}

void statistics_net(char *cfgfile, char *weightfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONNECTED && l.batch_normalize) {
            printf("Connected Layer %d\n", i);
            statistics_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize) {
            printf("GRU Layer %d\n", i);
            printf("Input Z\n");
            statistics_connected_layer(*l.input_z_layer);
            printf("Input R\n");
            statistics_connected_layer(*l.input_r_layer);
            printf("Input H\n");
            statistics_connected_layer(*l.input_h_layer);
            printf("State Z\n");
            statistics_connected_layer(*l.state_z_layer);
            printf("State R\n");
            statistics_connected_layer(*l.state_r_layer);
            printf("State H\n");
            statistics_connected_layer(*l.state_h_layer);
        }
        if (l.type == LSTM && l.batch_normalize) {
            printf("LSTM Layer %d\n", i);
            printf("wf\n");
            statistics_connected_layer(*l.wf);
            printf("wi\n");
            statistics_connected_layer(*l.wi);
            printf("wg\n");
            statistics_connected_layer(*l.wg);
            printf("wo\n");
            statistics_connected_layer(*l.wo);
            printf("uf\n");
            statistics_connected_layer(*l.uf);
            printf("ui\n");
            statistics_connected_layer(*l.ui);
            printf("ug\n");
            statistics_connected_layer(*l.ug);
            printf("uo\n");
            statistics_connected_layer(*l.uo);
        }
        printf("\n");
    }
}

void denormalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL && l.batch_normalize) {
            denormalize_convolutional_layer(l);
            net.layers[i].batch_normalize=0;
        }
        if (l.type == CONNECTED && l.batch_normalize) {
            denormalize_connected_layer(l);
            net.layers[i].batch_normalize=0;
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
            l.input_z_layer->batch_normalize = 0;
            l.input_r_layer->batch_normalize = 0;
            l.input_h_layer->batch_normalize = 0;
            l.state_z_layer->batch_normalize = 0;
            l.state_r_layer->batch_normalize = 0;
            l.state_h_layer->batch_normalize = 0;
            net.layers[i].batch_normalize=0;
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.wf);
            denormalize_connected_layer(*l.wi);
            denormalize_connected_layer(*l.wg);
            denormalize_connected_layer(*l.wo);
            denormalize_connected_layer(*l.uf);
            denormalize_connected_layer(*l.ui);
            denormalize_connected_layer(*l.ug);
            denormalize_connected_layer(*l.uo);
            l.wf->batch_normalize = 0;
            l.wi->batch_normalize = 0;
            l.wg->batch_normalize = 0;
            l.wo->batch_normalize = 0;
            l.uf->batch_normalize = 0;
            l.ui->batch_normalize = 0;
            l.ug->batch_normalize = 0;
            l.uo->batch_normalize = 0;
            net.layers[i].batch_normalize=0;
		}
    }
    save_weights(net, outfile);
}

void visualize(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    visualize_network(net);
#ifdef OPENCV
    wait_until_press_key_cv();
#endif
}

void* create_shared_memory(size_t size) {
  // Our memory buffer will be readable and writable:
  int protection = PROT_READ | PROT_WRITE;

  // The buffer will be shared (meaning other processes can access it), but
  // anonymous (meaning third-party processes cannot obtain an address for it),
  // so only this process and its children will be able to use it:
  int visibility = MAP_SHARED | MAP_ANONYMOUS;

  // The remaining parameters to `mmap()` are not important for this use case,
  // but the manpage for `mmap` explains their purpose.
  return mmap(NULL, size, protection, visibility, -1, 0);
}

// store resource configuration
// resource configuration shape 
// # of layers, 0 or 1, 0 or 1, ..... (0: cpu, 1: gpu)
int ** store_res_cfg(int res_cfg_num, char ** rpaths){

    char *buffer = NULL;
    char *tmp;
    int size;
    int layer_num;
    FILE *fp = NULL;
    int ** cfg_list = (int **)malloc(sizeof(int *)*res_cfg_num);
    for(int i =0; i < res_cfg_num; i ++){
        fp = fopen(rpaths[i], "r");

        //read size of file
        fseek(fp, 0, SEEK_END);
        size = ftell(fp);

        buffer = (char *)realloc(buffer,size + 1);

        fseek(fp, 0, SEEK_SET);
        fread(buffer, size, 1, fp);
        fclose(fp);
       
        tmp = strtok(buffer, ",");
        layer_num = atoi(tmp);
        cfg_list[i]  = (int *)malloc(sizeof(int)*layer_num);
        for(int j= 0; j < layer_num; j++){
            cfg_list[i][j]=atoi(strtok(NULL,","));
        }
        // set last layer on CPU 
    }
    return cfg_list;
}

//////////////// GPU ACESSING MANAGEING ////////////////
void swap(int *xp, int *yp)
{
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void bubbleSort(int arr[], int n)
{
   int i, j;
   for (i = 0; i < n-1; i++)      
 
       // Last i elements are already in place   
       for (j = 0; j < n-i-1; j++) 
           if (arr[j] < arr[j+1])
              swap(&arr[j], &arr[j+1]);
}

void enqueue(int* q, int val)
{
  for (int i=0; i<N; i++){
	if (q[i] == 0){
	q[i] = val;
	break;
	}
  }  
}

int dequeue(int* q)
{
	/* sort */
	bubbleSort(q, N);	

	int tmp =  q[0];
	for (int i=0; q[i]>0; i++){		
	q[i] = q[i+1];
	}

	return tmp;
}

void get_task_info(char * mytask, char ** argv){
    list * info_list = get_paths(mytask);
    char ** info = (char **)list_to_array(info_list);
    int info_num = info_list->size;
    if (0 == strcmp(info[0],"rnn")){
        if (info_num != 5){
            printf("[%d task] Something wrong with task info\n",identifier);
            exit(-1);
        }

        argv[1] = info[0];
        argv[2] = info[1];
        argv[3] = info[2];
        argv[4] = info[3];
        argv[5] = info[4];
    }
    else{
        if (info_num != 7){
            printf("[%d task] Something wrong with task info\n",identifier);
            exit(-1);
        }
        // task type (e.g classification, dection, nlp...)
        argv[1] = info[0];
        // submodule of task
        argv[2] = info[1];
        // label of data
        argv[3] = info[2];
        // model (e.g vgg, yolo, resnet ...)
        argv[4] = info[3];
        // weight 
        argv[5] = info[4];
        // period
        argv[6] = info[5];
        // input image
        argv[7] = info[6];
    }
}

int main(int argc, char **argv)
{
#ifdef _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
    _g_detector_params = (DetectorParams*)malloc(sizeof(DetectorParams));
    int j;
    int i;
    int k;
    
    for (i = 0; i < argc; ++i) {
		if (!argv[i]) continue;
		strip_args(argv[i]);
	}

    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    gpu_index = find_int_arg(argc, argv, "-i", 0);
    if(find_arg(argc, argv, "-nogpu")) {
        gpu_index = -1;
        printf("\n Currently Darknet doesn't support -nogpu flag. If you want to use CPU - please compile Darknet with GPU=0 in the Makefile, or compile darknet_no_gpu.sln on Windows.\n");
        exit(-1);
    }

    //////// GET PROCESS NUMBER /////////
    int process_num = find_int_arg(argc, argv, "-process_num", 1);
    int pid;    
    N = process_num;

    char * taskset = find_char_arg(argc, argv, "-taskset", 0);
    if (taskset == 0){
        puts("No information about taskset");
        exit(-1);
    }

    list *tlist = get_paths(taskset);
    char **tpaths = (char **)list_to_array(tlist);
    int task_num = tlist->size;

    if( process_num != task_num){
        puts("Task number and process number doesn't mach!\n");
        exit(-1);
    }

    ///////// GET RESOURCE CONFIGURATIONS /////////
    char * res_cfg = find_char_arg(argc,argv,"-resource",0);
    if(res_cfg == 0){
        printf("No information about resource configuration\n");
        exit(-1);
    }
    
    list *rlist = get_paths(res_cfg);
    char **rpaths = (char **)list_to_array(rlist);
    int res_cfg_num = rlist->size;

    if(process_num != res_cfg_num){
        printf("Resource configuration number and process number doesn't mach!\n");
        exit(-1);
    }

    // //////////// GET MODEL CONFIGURATIONS ////////////
    // char * model_list = find_char_arg(argc,argv,"-models",0);
    // if(model_list == 0){
    //     printf("No information about models\n");
    //     exit(-1);
    // }

    // list *mlist = get_paths(model_list);
    // char **mpaths = (char **)list_to_array(mlist);
    // int model_list_number = mlist->size;

    // if(process_num != model_list_number){
    //     printf("Model lists and process number doesn't mach!\n");
    //     exit(-1);
    // }

    // /////////// GET WEIGHT CONFIGURATIONS ////////////
    // char * weight_list = find_char_arg(argc,argv,"-weights",0);
    // if(weight_list == 0){
    //     printf("No information about weights\n");
    //     exit(-1);
    // }

    // list *wlist = get_paths(weight_list);
    // char **wpaths = (char **)list_to_array(wlist);
    // int weight_list_number = wlist->size;

    // if(process_num != weight_list_number){
    //     printf("Weight lists and process number doesn't mach!\n");
    //     exit(-1);        
    // }

    // /////////// GET DATA CONFIGURATIONS ////////////
    // char * data_list = find_char_arg(argc,argv,"-data",0);
    // if(data_list == 0){
    //     printf("No information about datas\n");
    //     exit(-1);
    // }

    // list *dlist = get_paths(data_list);
    // char **dpaths = (char **)list_to_array(dlist);
    // int data_list_number = wlist->size;

    // if(process_num != data_list_number){
    //     printf("Data lists and process number doesn't mach!\n");
    //     exit(-1);        
    // }

    /////// DEFINE SHARED MEMORY ////////
    shmem_rescfg = (int **)create_shared_memory(sizeof(int **));
    shmem_pid = (int *)create_shared_memory(sizeof(int)*process_num);
    shmem_timer = (struct timespec *)create_shared_memory(sizeof(struct timespec));
    
    // init shared memory
    shmem_rescfg = store_res_cfg(res_cfg_num,rpaths);
    memset(shmem_pid,0, sizeof(int)*process_num);

    int queue_id, mutex_id;
    int mode = S_IRWXU | S_IRWXG;
    
    mutex_id = shm_open(MYMUTEX, O_CREAT | O_RDWR | O_TRUNC, mode);
    if (mutex_id < 0) {
        perror("shm_open failed with " MYMUTEX);
        return -1;
    }
    if (ftruncate(mutex_id, sizeof(pthread_mutex_t)) == -1) {
        perror("ftruncate failed with " MYMUTEX);
        return -1;
    }
    gpu_lock = (pthread_mutex_t *)mmap(NULL, sizeof(pthread_mutex_t), PROT_READ | PROT_WRITE, MAP_SHARED, mutex_id, 0);
    if (gpu_lock == MAP_FAILED) {
        perror("mmap failed with " MYMUTEX);
        return -1;
    }
    /* cond */
    queue_id = shm_open(MYQUEUE, O_CREAT | O_RDWR | O_TRUNC, mode);
    if (queue_id < 0) {
        perror("shm_open failed with " MYQUEUE);
        return -1;
    }
    if (ftruncate(queue_id, sizeof(int)) == -1) {
        perror("ftruncate failed with " MYQUEUE);
        return -1;
    }
    queue = (int *)mmap(NULL, N*sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED, queue_id, 0);
    if (queue == MAP_FAILED) {
        perror("ftruncate failed with " MYQUEUE);
        return -1;
    }


    // gpu_lock = (pthread_mutex_t *)create_shared_memory(sizeof(pthread_mutex_t));
    // queue = (int *)create_shared_memory(sizeof(int)*process_num);
    /* set mutex shared between processes */
    pthread_mutexattr_t mattr;
    pthread_mutexattr_init(&mattr);
    pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(gpu_lock, &mattr);
    pthread_mutexattr_destroy(&mattr);
   

    if(process_num){
        pid = fork();
        if(!pid) { /*if child*/ identifier = 0; }
        for(int i = 0; i < process_num-1 ; i++){
            if(pid) { /* if mother */
                pid = fork();
                if(!pid) { identifier = i+1;}
            }
        }
        printf("my pid: %d, my_identifier: %d\n",getpid(),identifier);
    }

    if(identifier == -1){ /* mother process */ 

        sleep(20);
        for (int i = 0; i < process_num; i++){
            kill(shmem_pid[i],SIGCONT);
        }
        
        wait();
    }else{ /* child process */
#ifndef GPU
        gpu_index = -1;
#else
        if(gpu_index >= 0){
            cuda_set_device(gpu_index);
            CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
        }
#endif
        // CPU Affinity setting //
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(0, &mask);
        sched_setaffinity(0,sizeof(mask), &mask);

        //set CPU execution priority.
        setpriority(PRIO_PROCESS, getpid(), -10-identifier);
        char * mytask = tpaths[identifier];
        
        //allocate resource configuration of each process.

        test_extern_arr = shmem_rescfg[identifier];
        get_task_info(mytask, argv);
        ///// data cfg
        printf("[%d], data path %s\n",identifier, argv[3]);
        ///// model cfg
        printf("[%d], model path %s\n",identifier, argv[4]);
        ///// weight cfg
        printf("[%d], weight path %s\n",identifier, argv[5]);
        //redirect stdout & stderr to certain file.
        int fd;                 //fd
        char output_idx[4];     //idx of output file
        char output_name[100];   //name of output file
        char file_extension[20];
        
        sprintf(file_extension,"%s",".txt");
        sprintf(output_idx,"%d",identifier);
        strcpy(output_name,"logs/multi_log/multiprocresult_");
        strcat(output_name,output_idx);
        strcat(output_name,file_extension);
        if((fd = open(output_name, O_RDWR | O_CREAT,0666))==-1){
            perror("open");
            return 1;
        }
        dup2(fd,STDOUT_FILENO);
        dup2(fd,STDERR_FILENO);
        close(fd);
        //!stdout redirection.
        printf("My pid: %d, my_identifier: %d\n",getpid(),identifier);
        //original darknet main process.
        if (0 == strcmp(argv[1], "average")){
            average(argc, argv);
        } else if (0 == strcmp(argv[1], "yolo")){
            run_yolo(argc, argv);
        } else if (0 == strcmp(argv[1], "voxel")){
            run_voxel(argc, argv);
        } else if (0 == strcmp(argv[1], "super")){
            run_super(argc, argv);
        } else if (0 == strcmp(argv[1], "detector")){
            run_detector(argc, argv);
        } else if (0 == strcmp(argv[1], "detect")){
            float thresh = find_float_arg(argc, argv, "-thresh", .24);
            int ext_output = find_arg(argc, argv, "-ext_output");
            char *filename = (argc > 4) ? argv[4]: 0;
            test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, 0.5, 0, ext_output, 0, NULL, 0, 0);
        } else if (0 == strcmp(argv[1], "cifar")){
            run_cifar(argc, argv);
        } else if (0 == strcmp(argv[1], "go")){
            run_go(argc, argv);
        } else if (0 == strcmp(argv[1], "rnn")){
            run_char_rnn(argc, argv);
        } else if (0 == strcmp(argv[1], "vid")){
            run_vid_rnn(argc, argv);
        } else if (0 == strcmp(argv[1], "coco")){
            run_coco(argc, argv);
        } else if (0 == strcmp(argv[1], "classify")){
            predict_classifier("cfg/imagenet1k.data", argv[2], argv[3], argv[4], 5);
        } else if (0 == strcmp(argv[1], "classifier")){
            run_classifier(argc, argv);
        } else if (0 == strcmp(argv[1], "art")){
            run_art(argc, argv);
        } else if (0 == strcmp(argv[1], "tag")){
            run_tag(argc, argv);
        } else if (0 == strcmp(argv[1], "compare")){
            run_compare(argc, argv);
        } else if (0 == strcmp(argv[1], "dice")){
            run_dice(argc, argv);
        } else if (0 == strcmp(argv[1], "writing")){
            run_writing(argc, argv);
        } else if (0 == strcmp(argv[1], "3d")){
            composite_3d(argv[2], argv[3], argv[4], (argc > 5) ? atof(argv[5]) : 0);
        } else if (0 == strcmp(argv[1], "test")){
            test_resize(argv[2]);
        } else if (0 == strcmp(argv[1], "captcha")){
            run_captcha(argc, argv);
        } else if (0 == strcmp(argv[1], "nightmare")){
            run_nightmare(argc, argv);
        } else if (0 == strcmp(argv[1], "rgbgr")){
            rgbgr_net(argv[2], argv[3], argv[4]);
        } else if (0 == strcmp(argv[1], "reset")){
            reset_normalize_net(argv[2], argv[3], argv[4]);
        } else if (0 == strcmp(argv[1], "denormalize")){
            denormalize_net(argv[2], argv[3], argv[4]);
        } else if (0 == strcmp(argv[1], "statistics")){
            statistics_net(argv[2], argv[3]);
        } else if (0 == strcmp(argv[1], "normalize")){
            normalize_net(argv[2], argv[3], argv[4]);
        } else if (0 == strcmp(argv[1], "rescale")){
            rescale_net(argv[2], argv[3], argv[4]);
        } else if (0 == strcmp(argv[1], "ops")){
            operations(argv[2]);
        } else if (0 == strcmp(argv[1], "speed")){
            speed(argv[2], (argc > 3 && argv[3]) ? atoi(argv[3]) : 0);
        } else if (0 == strcmp(argv[1], "oneoff")){
            oneoff(argv[2], argv[3], argv[4]);
        } else if (0 == strcmp(argv[1], "partial")){
            partial(argv[2], argv[3], argv[4], atoi(argv[5]));
        } else if (0 == strcmp(argv[1], "average")){
            average(argc, argv);
        } else if (0 == strcmp(argv[1], "visualize")){
            visualize(argv[2], (argc > 3) ? argv[3] : 0);
        } else if (0 == strcmp(argv[1], "imtest")){
            test_resize(argv[2]);
        } else if (0 == strcmp(argv[1], "thread")){
            multi_thread(argc,argv);
        } else if (0 == strcmp(argv[1], "deepdive")){
            deep_dive(argc,argv);
        } else {
            fprintf(stderr, "Not an option: %s\n", argv[1]);
        }
        free(_g_detector_params);
        return 0;
    }
}
