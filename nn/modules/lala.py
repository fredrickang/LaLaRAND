from collections import OrderedDict
import math
import torch
from torch.nn.parameter import Parameter
from .. import functional as F
from .. import init
from .module import Module
from .utils import _single, _pair, _triple
from ..._jit_internal import List
import torch.utils.hooks as hooks
import copy
import time

def find_wrapper_by_idx(wrapper_list, idx):
    for wrapper in wrapper_list:
        if wrapper._layer_idx == idx:
            return wrapper
    return None

def generate_wrapper(mod, run_fn, run_args):
    wrapper_list = []
    
    mod = mod.cuda() 
    mod.count_layers()
    
    for name, module in mod.named_modules():
        if module._get_name() in mod._layer_names:
            tmp = copy.deepcopy(module)
            wrapper = torch.nn.modules.Wrapper(tmp._get_name())
            wrapper._layer_idx = tmp._layer_idx
            wrapper.setGPUModule(tmp)
            wrapper_list.append(wrapper)

    mod = mod.cpu()
    
    for name, module in mod.named_modules():
        if module._get_name() in mod._layer_names:
            tmp = copy.deepcopy(module)
            wrapper = find_wrapper_by_idx(wrapper_list, tmp._layer_idx)
            wrapper.setCPUModule(tmp)
    
    ## Input calibration for quantization 
    mod = torch.quantization.prepare_lala(mod, inplace=False)
    run_fn(mod, run_args[0], run_args[1])
    
    ## Collect Quatizer per layer
    for name, module in mod.named_modules():
        if module._get_name() in mod._layer_names:
           wrapper = find_wrapper_by_idx(wrapper_list, module._layer_idx)
           wrapper.setQuantizer(torch.nn.quantized.Quantize.from_float(module))

    ## Convert into Quantized Model
    mod = torch.quantization.convert(mod, inplace=False)
    for name, module in mod.named_modules():
        if module._get_name() == "MaxPool2d" or module._get_name() == "AdaptiveAvgPool2d":
            module.clear_forward_pre_hook()
    
    ## Register Quantized Modules
    mod.count_layers()
    for name, module in mod.named_modules():
        if module._get_name() in mod._layer_names:
            wrapper = find_wrapper_by_idx(wrapper_list, module._layer_idx)
            wrapper.setQuantModule(module)
    return wrapper_list

def swap_modules(mod, wrapper_list):
    for name, module in mod.named_modules():
        for name2, module2 in module.named_children():
            if name2 == 'cpu_module' or name2 == 'gpu_module' or name2 == 'quant_module':
                continue
            if module2._get_name() in mod._layer_names:
                module._modules[name2] = find_wrapper_by_idx(wrapper_list, module2._layer_idx)

def find_module_by_layer_num(mod, layer_num):
    for name, module in mod.named_modules():
        if module._get_name() == "Wrapper" and module.layer_num == layer_num :
            return module

def setRuntimeQuantizer(mod):
    for name, module in mod.named_modules():
        if module._get_name() == "Wrapper":
           if module.layer_num != mod._nlayers-1 and module.layer_num != -1:
               next_module = find_module_by_layer_num(mod, module.layer_num +1)
               module.setRuntimeQuantModule(find_module_by_layer_num(mod, module.layer_num +1).quantizer)
           if module.layer_num == mod._nlayers-1:
               module.setRuntimeQuantModule(torch.nn.Identity())

class Wrapper(Module):
    def __init__(self, name):
        super(Wrapper, self).__init__()       
        self.name = name
        self.gpu_module = None
        self.cpu_module = None
        self.quant_module = None
        self.dequant_module = torch.nn.quantized.DeQuantize()
        self.quantizer = None
        self.floatfunc_pre_hook = OrderedDict()
        self.floatfunc_hook = OrderedDict()

    def setGPUModule(self, mod):
        mod = copy.deepcopy(mod)
        self.gpu_module = mod
    
    def setCPUModule(self, mod):
        mod = copy.deepcopy(mod)
        self.cpu_module = mod
    
    def setQuantModule(self, mod):
        self.quant_module = mod
    
    def setQuantizer(self, quant):
        quant = copy.deepcopy(quant)
        self.quantizer = quant
    
    def register_floatfunc_pre_hook(self, hook):
        handle = hooks.RemovableHandle(self.floatfunc_pre_hook)
        self.floatfunc_pre_hook[handle.id] = hook
        return handle
    
    def register_floatfunc_hook(self, hook):
        handle = hooks.RemovableHandle(self.floatfunc_hook)
        self.floatfunc_hook[handle.id] = hook
        return handle

    def _add(self, x, y):
        if self.resource == 0:    
            if x.is_cuda:
                x = x.cpu()

            if (x.dtype != torch.qint8 and x.dtype != torch.quint8):
                x = self.quantizer(x)

            if y.is_cuda:
                y = y.cpu()
            
            if (y.dtype != torch.qint8 and y.dtype != torch.quint8):
                y = self.quantizer(y)    
            
            return self.quant_module.add(x, y)
        
        if self.resource == 2 or self.resource ==3:
            if not x.is_cuda:
                if x.dtype == torch.qint8 or x.dtype == torch.quint8:
                    x = self.dequant_module(x)
                x = x.cuda()

            if not y.is_cuda:
                if y.dtype == torch.qint8 or y.dtype == torch.quint8:
                    y = self.dequant_module(y)
                y = y.cuda()           

#           print("Executing GPU Module")
            return self.gpu_module.add(x,y)
        
        if self.resource == 1:
            if x.is_cuda:
                x = x.cpu()
            if y.is_cuda:
                y = y.cpu()
            return self.cpu_module.add(x,y)

    def add(self, x, y):
        for hook in self.floatfunc_pre_hook.values():
            result = hook(self, x)

        result = self._add(x, y)
        
        for hook in self.floatfunc_hook.values():
            hook(self, x, result)
        
        return result
    
    def _cat(self, x, dim=0):
        new_x = tuple()
        
        if self.resource == 0:
            for tk in x:
                if tk.is_cuda:
                    tk = tk.cpu()

                if (tk.dtype != torch.qint8 and tk.dtype != torch.quint8):
                    tk = self.quantizer(tk)
                new_x += (tk,)
            return self.quant_module.cat(new_x, dim)

        if self.resource == 2 or self.resource == 3:
            for tk in x:
                if not tk.is_cuda:
                    if tk.dtype == torch.qint8 or tk.dtype == torch.quint8:
                        tk = self.dequant_module(tk)
                    tk = tk.cuda()

                new_x += (tk,)
            return self.gpu_module.cat(new_x, dim)
        
        if self.resource == 1:
            for tk in x:
                if tk.is_cuda:
                    tk = tk.cpu()
                new_x += (tk,)
            return self.cpu_module.cat(new_x, dim);

    def cat(self, x, dim=0):
        for hook in self.floatfunc_pre_hook.values():
            result = hook(self, input)

        result = self._cat(x, dim)
        for hook in self.floatfunc_hook.values():
            hook(self, input, result)

        return result

    def forward(self, input):
        resource = self.resource ## 2:GPU 1:CPU 0:QUANT

        require_quant = False
        require_dequant = False
        
        if (input.dtype == torch.qint8 or input.dtype == torch.quint8) and (resource == 2 or resource == 3):
            require_dequant = True
        if (input.dtype != torch.qint8 and input.dtype != torch.quint8) and resource == 0:
            require_quant = True
        
        if require_dequant:
            input = self.dequant_module(input)
            input = input.cuda()
        
        if require_quant:
            if input.is_cuda:
                input = input.cpu()
            
            input = self.quantizer(input)

        if not input.is_cuda and (resource == 2 or resource ==3):
            input = input.cuda()
        if input.is_cuda and resource == 1:
            input = input.cpu()

        if resource == 0:
            return self.quant_module(input)
        elif resource == 1:
            out = self.cpu_module(input)
            return out
        elif resource == 2:
            return self.gpu_module(input)
        else:
            assert("Wrong Config")



