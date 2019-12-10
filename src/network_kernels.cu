#include "dark_cuda.h"

#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>
#include <signal.h>
#include <sys/resource.h>

#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "parser.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "rnn_layer.h"
#include "gru_layer.h"
#include "crnn_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "cost_layer.h"
#include "local_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "blas.h"

//#ifdef OPENCV
//#include <opencv2/highgui/highgui_c.h>
//#endif

#include "http_stream.h"

float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float * get_network_output_gpu(network net);
void enqueue(int* q, int val);
int dequeue(int* q);

extern int* test_extern_arr;
extern int identifier;
extern int * queue;
extern pthread_mutex_t *gpu_lock;
extern int N;

void forward_network_gpu(network net, network_state state)
{
    cudaDeviceSynchronize();
    //printf("\n");
    state.workspace = net.workspace;
    state.workspace_cpu = net.workspace_cpu;
    int pid;
    int i;
    int *res_arr;
    double _time;
    double time;
    res_arr = test_extern_arr;
    for(i = 0; i < net.n; ++i){
        
        state.index = i;
        layer l = net.layers[i];
        
        if(l.delta_gpu && state.train){
            fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }   
        
        time  = get_time_point();
        
        if (res_arr[i] == 0){ // on cpu
//            printf("[%2dth] Excess  : %8.5f\n",i,get_time_point());
            if (l.type == CONVOLUTIONAL && net.quantized == 1 && l.index >=1 && l.activation != LINEAR) {
                l.forward_quant(l, state); // w/ quantize
            }
            else {
                l.forward(l,state);   //  w/o quantize
            }
//            printf("[%2dth] Finish  : %8.5f\n",i,get_time_point());
        }
        else{ // on gpu 
            //            printf("[%2dth] Request : %8.5f\n",i,get_time_point());
            // gpu access control by mutex
            while(pthread_mutex_trylock(gpu_lock)){
                //printf("[Process %d put into wait]\n", identifier);
                enqueue(queue, getpid());
                kill(getpid(), SIGSTOP);
                setpriority(PRIO_PROCESS, getpid(), -20);
                continue;
            }
//            printf("[%2dth] Excess  : %8.5f\n",i,get_time_point());
            l.forward_gpu(l, state);
            CHECK_CUDA(cudaDeviceSynchronize());

            setpriority(PRIO_PROCESS, getpid(), -10-identifier);
            
            pthread_mutex_unlock(gpu_lock);
//            printf("[%2dth] Finish  : %8.5f\n",i,get_time_point());
            kill( pid = dequeue(queue), SIGCONT);        
//            printf("pid: %d, has been waked\n",pid);
        }
        printf("[Process %d] layer: %3d type: %15s - Predicted in %8.5f milli-seconds.\n", identifier, i, get_layer_string(l.type), ((double)get_time_point() -time) / 1000);
        
        sleep(0.01);
        if(net.wait_stream)
            cudaStreamSynchronize(get_cuda_stream());

        if(res_arr[i] == 0){//currently running on CPU
            if(res_arr[i+1] == 0){//next is running on CPU
                state.input = l.output;    
            }
            else{//next is running on GPU
                _time = get_time_point();
                //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
                state.input = l.output;
            }
        }
        else{//currently running on GPU
            if(res_arr[i+1] == 0){//next is running on CPU
                _time = get_time_point();
                //cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
                state.input = l.output_gpu;
            }
            else{//next is running on GPU
                state.input = l.output_gpu;
            }
        }
    }
}

void backward_network_gpu(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    float * original_input = state.input;
    float * original_delta = state.delta;
    for(i = net.n-1; i >= 0; --i){
        state.index = i;
        layer l = net.layers[i];
        if (l.stopbackward) break;
        if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }else{
            layer prev = net.layers[i-1];
            state.input = prev.output_gpu;
            state.delta = prev.delta_gpu;
        }
        if (l.onlyforward) continue;
        l.backward_gpu(l, state);

        /*
        if(i != 0)
        {
            layer l = net.layers[i - 1];
            int state_delta_nan_inf = is_nan_or_inf(state.delta, l.outputs * l.batch);
            int state_input_nan_inf = is_nan_or_inf(state.input, l.outputs * l.batch);
            printf("\n i - %d  is_nan_or_inf(s.delta) = %d \n", i, state_delta_nan_inf);
            printf(" i - %d  is_nan_or_inf(s.input) = %d \n", i, state_input_nan_inf);
            if (state_delta_nan_inf || state_input_nan_inf) { printf(" found "); getchar(); }
        }
        */
    }
}

void update_network_gpu(network net)
{
    cuda_set_device(net.gpu_index);
    int i;
    int update_batch = net.batch*net.subdivisions * get_sequence_value(net);
    float rate = get_current_rate(net);
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        l.t = get_current_batch(net);
        if(l.update_gpu){
            l.update_gpu(l, update_batch, rate, net.momentum, net.decay);
        }
    }
}

void forward_backward_network_gpu(network net, float *x, float *y)
{
    network_state state;
    state.index = 0;
    state.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = get_network_output_size(net)*net.batch;
    if(net.layers[net.n-1].truths) y_size = net.layers[net.n-1].truths*net.batch;
    if(!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
        *net.truth_gpu = cuda_make_array(y, y_size);
    }else{
        cuda_push_array(*net.input_gpu, x, x_size);
        cuda_push_array(*net.truth_gpu, y, y_size);
    }
    state.input = *net.input_gpu;
    state.delta = 0;
    state.truth = *net.truth_gpu;
    state.train = 1;
#if defined(CUDNN_HALF) && defined(CUDNN)
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (net.cudnn_half){
            if (l.type == CONVOLUTIONAL && l.weights_gpu && l.weights_gpu16) {
                assert((l.nweights) > 0);
                cuda_convert_f32_to_f16(l.weights_gpu, l.nweights, l.weights_gpu16);
            }
            else if (l.type == CRNN && l.input_layer->weights_gpu && l.input_layer->weights_gpu16) {
                assert((l.input_layer->c*l.input_layer->n*l.input_layer->size*l.input_layer->size) > 0);
                cuda_convert_f32_to_f16(l.input_layer->weights_gpu, l.input_layer->nweights, l.input_layer->weights_gpu16);
                cuda_convert_f32_to_f16(l.self_layer->weights_gpu, l.self_layer->nweights, l.self_layer->weights_gpu16);
                cuda_convert_f32_to_f16(l.output_layer->weights_gpu, l.output_layer->nweights, l.output_layer->weights_gpu16);
            }
            else if (l.type == CONV_LSTM && l.wf->weights_gpu && l.wf->weights_gpu16) {
                assert((l.wf->c * l.wf->n * l.wf->size * l.wf->size) > 0);
                if (l.peephole) {
                    cuda_convert_f32_to_f16(l.vf->weights_gpu, l.vf->nweights, l.vf->weights_gpu16);
                    cuda_convert_f32_to_f16(l.vi->weights_gpu, l.vi->nweights, l.vi->weights_gpu16);
                    cuda_convert_f32_to_f16(l.vo->weights_gpu, l.vo->nweights, l.vo->weights_gpu16);
                }
                cuda_convert_f32_to_f16(l.wf->weights_gpu, l.wf->nweights, l.wf->weights_gpu16);
                cuda_convert_f32_to_f16(l.wi->weights_gpu, l.wi->nweights, l.wi->weights_gpu16);
                cuda_convert_f32_to_f16(l.wg->weights_gpu, l.wg->nweights, l.wg->weights_gpu16);
                cuda_convert_f32_to_f16(l.wo->weights_gpu, l.wo->nweights, l.wo->weights_gpu16);
                cuda_convert_f32_to_f16(l.uf->weights_gpu, l.uf->nweights, l.uf->weights_gpu16);
                cuda_convert_f32_to_f16(l.ui->weights_gpu, l.ui->nweights, l.ui->weights_gpu16);
                cuda_convert_f32_to_f16(l.ug->weights_gpu, l.ug->nweights, l.ug->weights_gpu16);
                cuda_convert_f32_to_f16(l.uo->weights_gpu, l.uo->nweights, l.uo->weights_gpu16);
            }
        }
    }
#endif
    forward_network_gpu(net, state);
    //cudaStreamSynchronize(get_cuda_stream());
    backward_network_gpu(net, state);
}

float train_network_datum_gpu(network net, float *x, float *y)
{
    *net.seen += net.batch;
    forward_backward_network_gpu(net, x, y);
    float error = get_network_cost(net);
    //if (((*net.seen) / net.batch) % net.subdivisions == 0) update_network_gpu(net);
    const int sequence = get_sequence_value(net);
    if (((*net.seen) / net.batch) % (net.subdivisions*sequence) == 0) update_network_gpu(net);

    return error;
}

typedef struct {
    network net;
    data d;
    float *err;
} train_args;

void *train_thread(void *ptr)
{
    train_args args = *(train_args*)ptr;
    free(ptr);
    cuda_set_device(args.net.gpu_index);
    *args.err = train_network(args.net, args.d);
    return 0;
}

pthread_t train_network_in_thread(network net, data d, float *err)
{
    pthread_t thread;
    train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
    ptr->net = net;
    ptr->d = d;
    ptr->err = err;
    if(pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed");
    return thread;
}

void pull_updates(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
        cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
        if(l.scale_updates) cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
        cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_updates(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
        cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
        if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
        cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void update_layer(layer l, network net)
{
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    l.t = get_current_batch(net);
    if(l.update_gpu){
        l.update_gpu(l, update_batch, rate, net.momentum, net.decay);
    }
}

void merge_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.biases, 1, base.biases, 1);
        axpy_cpu(l.nweights, 1, l.weights, 1, base.weights, 1);
        if (l.scales) {
            axpy_cpu(l.n, 1, l.scales, 1, base.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.biases, 1, base.biases, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, base.weights, 1);
    }
}

void scale_weights(layer l, float s)
{
    if (l.type == CONVOLUTIONAL) {
        scal_cpu(l.n, s, l.biases, 1);
        scal_cpu(l.nweights, s, l.weights, 1);
        if (l.scales) {
            scal_cpu(l.n, s, l.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        scal_cpu(l.outputs, s, l.biases, 1);
        scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
    }
}


void pull_weights(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_pull_array(l.biases_gpu, l.biases, l.n);
        cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
        if(l.scales) cuda_pull_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
        cuda_pull_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void push_weights(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, l.biases, l.n);
        cuda_push_array(l.weights_gpu, l.weights, l.nweights);
        if(l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, l.biases, l.outputs);
        cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void distribute_weights(layer l, layer base)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, base.biases, l.n);
        cuda_push_array(l.weights_gpu, base.weights, l.nweights);
        if(base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, base.biases, l.outputs);
        cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
    }
}


void merge_updates(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
        axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weight_updates, 1);
        if (l.scale_updates) {
            axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weight_updates, 1);
    }
}

void distribute_updates(layer l, layer base)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
        cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.nweights);
        if(base.scale_updates) cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
        cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
    }
}

void sync_layer(network *nets, int n, int j)
{
    //printf("Syncing layer %d\n", j);
    int i;
    network net = nets[0];
    layer base = net.layers[j];
    cuda_set_device(net.gpu_index);
    pull_weights(base);
    for (i = 1; i < n; ++i) {
        cuda_set_device(nets[i].gpu_index);
        layer l = nets[i].layers[j];
        pull_weights(l);
        merge_weights(l, base);
    }
    scale_weights(base, 1./n);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i].gpu_index);
        layer l = nets[i].layers[j];
        distribute_weights(l, base);
    }
    //printf("Done syncing layer %d\n", j);
}

typedef struct{
    network *nets;
    int n;
    int j;
} sync_args;

void *sync_layer_thread(void *ptr)
{
    sync_args args = *(sync_args*)ptr;
    sync_layer(args.nets, args.n, args.j);
    free(ptr);
    return 0;
}

pthread_t sync_layer_in_thread(network *nets, int n, int j)
{
    pthread_t thread;
    sync_args *ptr = (sync_args *)calloc(1, sizeof(sync_args));
    ptr->nets = nets;
    ptr->n = n;
    ptr->j = j;
    if(pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed");
    return thread;
}

void sync_nets(network *nets, int n, int interval)
{
    int j;
    int layers = nets[0].n;
    pthread_t *threads = (pthread_t *) calloc(layers, sizeof(pthread_t));

    *nets[0].seen += interval * (n-1) * nets[0].batch * nets[0].subdivisions;
    for (j = 0; j < n; ++j){
        *nets[j].seen = *nets[0].seen;
    }
    for (j = 0; j < layers; ++j) {
        threads[j] = sync_layer_in_thread(nets, n, j);
    }
    for (j = 0; j < layers; ++j) {
        pthread_join(threads[j], 0);
    }
    free(threads);
}

float train_networks(network *nets, int n, data d, int interval)
{
    int i;
#ifdef _DEBUG
    int batch = nets[0].batch;
    int subdivisions = nets[0].subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
#endif
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
    float *errors = (float *) calloc(n, sizeof(float));

    float sum = 0;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        threads[i] = train_network_in_thread(nets[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        //printf("%f\n", errors[i]);
        sum += errors[i];
    }
    //cudaDeviceSynchronize();
    if (get_current_batch(nets[0]) % interval == 0) {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(nets, n, interval);
        printf("Done!\n");
    }
    //cudaDeviceSynchronize();
    free(threads);
    free(errors);
    return (float)sum/(n);
}

float *get_network_output_layer_gpu(network net, int i)
{
    double _time = get_time_point();
    layer l = net.layers[i];
    if(l.type != REGION){
        //printf("l.type is %s\n",get_layer_string(l.type));
        //printf("test_extern_arr : %d\n",test_extern_arr[i]);
        if(test_extern_arr[i] == 1){//from gpu
            //printf("pulled from gpu.\n");
            cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
        }
    }

    //printf("end of get_net_output, time is %8.5f millisec\n",((double)get_time_point() - _time)/1000);
    return l.output;
}

float *get_network_output_gpu(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    //printf("target layer i is %d.\n",i);
    return get_network_output_layer_gpu(net, i);
}

float *network_predict_gpu(network net, float *input)
{
    int* res_arr;           // change the scope of memory according to resource allocation.
    float* temp_ptr[net.n]; // temporary pointers for cudaMalloc or malloc memories.
    double _time_cp;        // gpu_memcpy_timer.
    int i;

    double _time = get_time_point();
    if (net.gpu_index != cuda_get_device())
        cuda_set_device(net.gpu_index);
    int size = get_network_input_size(net) * net.batch;
    network_state state;
    state.index = 0;
    state.net = net;
    //state.input = cuda_make_array(input, size);   // memory will be allocated in the parse_network_cfg_custom() 
    
    res_arr = test_extern_arr;

    if (res_arr[0] == 0){//first network runs on cpu.
        memcpy(net.input_pinned_cpu, input, size*sizeof(float));
        state.input = net.input_pinned_cpu;
    //      printf("this is input%d\n",*state.input);
    }
    else{//first network runs on gpu.
        state.input = net.input_state_gpu;
        _time_cp = get_time_point();//init timer.
        memcpy(net.input_pinned_cpu, input, size * sizeof(float));
        cuda_push_array(state.input, net.input_pinned_cpu, size);
    }
    state.truth = 0;
    state.train = 0;
    state.delta = 0;

    //allocate unified cuda memories.
    //printf("start of unified memory reallocation\n");
    for(i = 0; i < net.n; ++i){
        if((res_arr[i] != res_arr[i+1]) || (i==8) || (i==13) || (i==19)){//computation resource change || route layer target.
            layer *lptr = &(net.layers[i]);
            if(res_arr[i] == 0){//if prev resource was CPU
                temp_ptr[i] = lptr->output;
                lptr->output = cuda_make_array_global(lptr->output,lptr->batch * lptr->outputs);
            }
            else{//if prev resource was GPU
                temp_ptr[i] = lptr->output_gpu;
                lptr->output_gpu = cuda_make_array_global(lptr->output,lptr->batch * lptr->outputs);
            }
        }
    }
    //printf("end of unified memory reallocation\n");
    //!allocated.

    forward_network_gpu(net, state);
    float *out = get_network_output_gpu(net);

    //free cuda memories and return original memory pointer.
    //printf("start of returning memory reallocation\n");
    for(i=0; i<net.n; ++i){
        if(res_arr[i] != res_arr[i+1]){
            layer *lptr = &(net.layers[i]);
            if(res_arr[i] == 0){
                cuda_free(lptr->output);
                lptr->output = temp_ptr[i];
            }
            else{
                cuda_free(lptr->output_gpu);
                lptr->output_gpu = temp_ptr[i];
            }
        }
    }
    //printf("end of returning memory reallocation\n");
    //!freed.

    //cuda_free(state.input);   // will be freed in the free_network()
    return out;
}


////////////// GPU ACCESSING MANIGNING //////////////
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
