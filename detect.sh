#./darknet detector periodic -models models.list -weights weights.list -data data.list -res_cfg resource.list -period 220 -process_num 4 -filename coco/100.part
#./darknet classifier predict ./cfg/imagenet1k.data ./extraction.cfg ./extraction.weights -res_cfg res_cfg.part -process_num 1 ./data/eagle.jpg
./darknet -taskset taskset.list -resource resource.list -process_num 1







#-thread_num 2 #< ./data/test_mult.txt

