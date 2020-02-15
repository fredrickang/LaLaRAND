#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>




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


dnn_info ** network_register(int register_fd, int dnns){
    
    int count =0;
    int err;
    // make storage
    dnn_info ** dnn_list = (dnn_info **)malloc(sizeof(dnn_info *)* dnns);
    for(int i =0 ; i < dnns ; i ++)
        dnn_list[i] = (dnn_info *)malloc(sizeof(dnn_info));
    
    //read
    while(count < dnns){
       if(err = read(register_fd, dnn_list[count], sizeof(dnn_list[count])) <0){
           puts("[ERROR]Fail to read register info");
           exit(-1);
       }
       //log
       puts("======================");
       printf("%d/%d",count+1, dnns);
       printf("[pid] %d, [layers] %d\n", dnn_list[count]->pid, dnn_list[count]->layer_num);
       puts("registered"); 
       count ++;
    }
    
    return dnn_list;
}



int main(int argc, char **argv){
    int dnns = find_int_arg(argc, argv, "-dnns", 1);
    int isSync = find_int_arg(argc, argv, "-sync", 1);
    
    int register_fd;
    
    dnn_info ** dnn_list;
    // open channel for register
    if( mkfifo("./lalarand_register",0666) == -1){
        puts("[ERROR]Fail to make pipe ");
        exit(-1);
    }

    if( (register_fd = open("./lalarand_register", O_RDWR)) < 0){
        puts("[ERROR]Fail to open channel for register");
        exit(-1);
    }
    else{
        puts("Channel for register has been successfully openned!"); 
    }
    
    // network register 
    puts("");
    printf("Waiting for %d dnns to be registered\n",dnns);
    dnn_list = network_register(register_fd, dnns);

}

