import os
import torch
import pynvml
import psutil
import numpy as np

# These are not working on Jetson devices

def get_all_gpu_utilisation():
    """Returns the memory usage for all visible GPUs."""
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return {}

    try:
        pynvml.nvmlInit()
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        num_gpus = torch.cuda.device_count()
        
        gpu_memory_info = {}
        
        for torch_gpu_id in range(num_gpus):
            if devices:
                devices_list = devices.split(",")
                if torch_gpu_id >= len(devices_list):
                    continue
                nvml_gpu_id = int(devices_list[torch_gpu_id])
            else:
                nvml_gpu_id = torch_gpu_id  # Direct mapping when no restrictions

            handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_gpu_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info_used = info.used // 1024**2  # Convert bytes to MB
            info_total = info.total // 1024**2
            gpu_memory_info[nvml_gpu_id] = (info_used, info_total)

        pynvml.nvmlShutdown()
        return gpu_memory_info

    except Exception as e:
        print(f"Error retrieving GPU utilization: {e}")
        return {}

def gpu_util(func):
    """Decorator to track GPU memory usage before and after function execution."""
    def fn_decorator(*args, **kwargs):
        before_usage = get_all_gpu_utilisation()
        result = func(*args, **kwargs)
        after_usage = get_all_gpu_utilisation()

        print("\n##############################")
        print("GPU Memory Usage Report:")
        for gpu_id in before_usage:
            before_used, total = before_usage[gpu_id]
            after_used, _ = after_usage.get(gpu_id, (before_used, total))  # Fallback in case of an error
            diff = after_used - before_used

            print(f"GPU {gpu_id}: Before: {before_used}/{total} MB | After: {after_used}/{total} MB | Difference: {diff} MB")
        
        print("\n")
        return result
    
    return fn_decorator


def get_cpu_utilisation():
    """Returns the CPU usage as a percentage."""
    cpu_percent = psutil.cpu_percent(interval=1)
    return cpu_percent

def gpu_cpu_util(func):
    """Decorator to track GPU and CPU memory usage before and after function execution."""
    def fn_decorator(*args, **kwargs):
        before_gpu_usage = get_all_gpu_utilisation()
        before_cpu_usage = get_cpu_utilisation()
        result = func(*args, **kwargs)
        after_gpu_usage = get_all_gpu_utilisation()
        after_cpu_usage = get_cpu_utilisation()

        print("\nGPU Memory Usage Report:")
        for gpu_id in before_gpu_usage:
            before_used, total = before_gpu_usage[gpu_id]
            after_used, _ = after_gpu_usage.get(gpu_id, (before_used, total))  # Fallback in case of an error
            diff = after_used - before_used

            print(f"GPU {gpu_id}: Before: {before_used}/{total} MB | After: {after_used}/{total} MB | Difference: {diff} MB")
        
        print("\nCPU Usage Report:")
        print(f"CPU Usage: Before: {before_cpu_usage}% | After: {after_cpu_usage}% | Difference: {after_cpu_usage - before_cpu_usage}%")
        print("##############################")
        print("\n")
        return result
    
    return fn_decorator
