# LaLaRAND: Flexible Layer-by-Layer CPU/GPU Scheduling for Real-Time DNN Tasks

LaLaRAND: Flexible Layer-by-Layer CPU/GPU Scheduling for Real-Time DNN Tasks <br>
Author: Woosung Kang, Kilho Lee, Jinkyu Lee, Insik Shin, Hoon Sung Chwa <br>
In 42nd IEEE Real-Time Systems Symposium (RTSS 2021) Dortmund, Germany, December 2021 <br>

## Requirements
CUDA: >= 10.2 <br>
cuDNN: >=8.0.2 <br>
PyTorch: 1.4.0 <br>
Python: >= 3.6 <br>
CMake: >= 3.10.2 <br>

## How to use
### PyTorch modification
1. Install PyTorch with version 1.4.0
2. Go to installation directory (probably /home/{username}/.local/lib/python{version}/site-packages/torch}
3. Replace directory nn, quantization.

### Scheduler
1. Run scheduler before DNN tasks
2. Provide resource configuration of DNN tasks by txt file (current: /tmp/{pid of task}.txt)

### DNN task
1. Before inference code, 
2. Call {model}.set_rt() to set rt-priority of task
3. Call {model}.hetero() to use heterogeous resource allocation
4. hetero() requires inference function and sample inputs for input calibaration 
