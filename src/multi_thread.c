#include "darknet.h"
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "http_stream.h"
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <assert.h>

int check_mistakes_mult = 0;

extern void run_detector(int argc, char **argv);

struct arg_info{
    int argc;
    char **argv;
};

struct deepdive{
    network net;
    image sized;
};

// MAKE DOBY WORKS
void * DOBY_IS_SLAVE(void * argument)
{
    struct arg_info *arg_info = argument;
    run_detector(arg_info->argc,arg_info->argv);
}


void multi_thread(int argc, char **argv)
{
    int thread_num = find_int_arg(argc, argv, "-thread_num",-1);
    int i;
    pthread_t tid;

    struct arg_info *arg_info = malloc(sizeof(struct arg_info));

    arg_info->argc = argc;
    arg_info->argv = argv;

    assert(thread_num != -1);
    for(i = 0; i< thread_num; i++){
        if(pthread_create(&tid, NULL,DOBY_IS_SLAVE, arg_info)) error("DOBBY IS FREEEEE\n");
    }
    
    pthread_join(tid, NULL);


}

void *DEEPDIVING(void * argument)
{
    struct deepdive *deepdive = argument;
    float *X = deepdive->sized.data;

    pid_t tid = getpid();

    double time = get_time_point();
    network_predict(deepdive->net, X);
    printf("%dth Doby: Predicted in %lf milli-seconds.\n", tid, ((double)get_time_point() - time) / 1000);

}

void deep_dive(int argc, char **argv)
{
    int dont_show = find_arg(argc, argv, "-dont_show");
    int show = find_arg(argc, argv, "-show");
    int letter_box = find_arg(argc, argv, "-letter_box");
    int calc_map = find_arg(argc, argv, "-map");
    int map_points = find_int_arg(argc, argv, "-points", 0);
    check_mistakes_mult = find_arg(argc, argv, "-check_mistakes");
    int show_imgs = find_arg(argc, argv, "-show_imgs");
    int mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", -1);
    int json_port = find_int_arg(argc, argv, "-json_port", -1);
    char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .25);    // 0.24
    float iou_thresh = find_float_arg(argc, argv, "-iou_thresh", .5);    // 0.5 for mAP
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int num_of_clusters = find_int_arg(argc, argv, "-num_of_clusters", 5);
    int width = find_int_arg(argc, argv, "-width", -1);
    int height = find_int_arg(argc, argv, "-height", -1);
    // extended output in test mode (output of rect bound coords)
    // and for recall mode (extended output table-like format with results for best_class fit)
    int ext_output = find_arg(argc, argv, "-ext_output");
    int save_labels = find_arg(argc, argv, "-save_labels");
    int thread_num = find_int_arg(argc, argv, "-thread_num",-1);
    if (argc < 4) {
        fprintf(stderr, "usage: %s %s [train/test/valid/demo/map] [data] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if (gpu_list) {
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = (int*)calloc(ngpus, sizeof(int));
        for (i = 0; i < ngpus; ++i) {
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',') + 1;
        }
    }
    else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    if (weights)
        if (strlen(weights) > 0)
            if (weights[strlen(weights) - 1] == 0x0d) weights[strlen(weights) - 1] = 0;
    char *filename = (argc > 6) ? argv[6] : 0;
    //if (0 == strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile, letter_box);
    
    char *cfgfile = cfg;
    char *weightfile = weights;

    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
    network net2;

    printf("copying network info...\n");
    memcpy(&net2, &net, sizeof(network));
    printf("done! result: %d %d\n",net.inputs,net2.inputs);
    
    if (weightfile) {
        load_weights(&net, weightfile);
    }
  // Our approach should not use layer fusion  
  // fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf(" Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        if (net.layers[net.n - 1].classes > names_size) getchar();
    }
    srand(2222222);
    char buff[256];
    char *input = buff;
    char *json_buf = NULL;
    int json_image_id = 0;
    FILE* json_file = NULL;
    if (outfile) {
        json_file = fopen(outfile, "wb");
        char *tmp = "[\n";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
    }
    int j;
    float nms = .45;    // 0.4F
    while (1) {
        if (filename) {
            strncpy(input, filename, 256);
            if (strlen(input) > 0)
                if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        }
        else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) break;
            strtok(input, "\n");
        }
        //image im;
        //image sized = load_image_resize(input, net.w, net.h, net.c, &im);
        image im = load_image(input, 0, 0, net.c);
        image sized;
        if(letter_box) sized = letterbox_image(im, net.w, net.h);
        else sized = resize_image(im, net.w, net.h);
        layer l = net.layers[net.n - 1];

        //box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        //float **probs = calloc(l.w*l.h*l.n, sizeof(float*));
        //for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)calloc(l.classes, sizeof(float));

        /// DEEP DIVE START
        pthread_t tid;
        struct deepdive *deepdive = malloc(sizeof(struct deepdive));
        struct deepdive *deepdive2 = malloc(sizeof(struct deepdive));
        net.t_idx = 1; 
        net2.t_idx = 2;
        deepdive->net = net;
        deepdive2->net = net2; 
        deepdive->sized = sized;
        deepdive2->sized = sized;
        int i;
        assert(thread_num != -1);
        for(i = 0; i< thread_num; i++){
            if (i%2 == 0){
                printf("create net1 thread.\n");
                if(pthread_create(&tid,NULL,DEEPDIVING, deepdive)) error("DEEPDIVE FAILED");
            }
            else{
                printf("create net2 thread.\n");
                if(pthread_create(&tid,NULL,DEEPDIVING, deepdive2)) error("DEEPDIVE FAILED");
            }
        }

        pthread_join(tid, NULL);

        free_image(im);
        free_image(sized);

        if (filename) break;
    }


    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    int i;
    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);

    free_network(net);
    //free_network(net2);
}

