"""
This file is used to meseaure min-area-rec operator, which will show how much improvment it has.
Usage
min_area_rect_torch_min_area_rect(Tensor pointsets, int N, Tensor result)
    Params:
        pointsets: -> Tensor [N, 18], N is the number of points
        N: -> int number of points
    Return:
        result -> Tensor [N, 8]
"""
import torch
import min_area_rect
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from cpu_operator.min_area_rect_cpu import min_area_rect_cpu
import seaborn as sns
import pandas as pd
import math
sns.set_theme(style="darkgrid")

def order_points(pts):
    ''' sort rectangle points by clockwise '''
    sort_x = pts[np.argsort(pts[:, 0]), :]
    
    Left = sort_x[:2, :]
    Right = sort_x[2:, :]
    # Left sort
    Left = Left[np.argsort(Left[:,1])[::-1], :]
    # Right sort
    Right = Right[np.argsort(Right[:,1]), :]
    
    return np.concatenate((Left, Right), axis=0)


def sort_result(arr):
    """
    sort numpy array according to order(x_min,y_min):
    """
    result = np.zeros((arr.shape[0],4,2))
    for i in range(arr.shape[0]):
        ret = arr[i].reshape(-1, 2)
        # ret = np.sort(ret, axis=0)
        ret = order_points(ret)
        result[i,:,:] = ret
    print(result)
    return result

def visualize(pointsets, result):
    plt.plot(pointsets[::2], pointsets[1::2], 'ro')
    result = np.insert(result, 0, result[-2:], axis=0)
    plt.plot(result[::2], result[1::2], 'b')

if __name__=="__main__":
    # extract args
    parser = argparse.ArgumentParser(description='Process arguments provided by user.')
    parser.add_argument('--num_test', action='store_const',
                    const=int,
                    help='the number of test')


    N = 9 # number of points
    # test using the simplest example
    num_pointsets = [1,3,5,10,18,100,1000, 2000, 10000, 50000, 100000]
    device = torch.device('cuda:0')
    gpu_time = 0 # recording start
    cpu_time = 0

    gpu_record = []
    cpu_record = []
    for num_pointset in num_pointsets:
        # GPU operator test
        simplest_data = np.random.random([num_pointset ,2*N]) # [1, 18] 
        simplest_data_gpu = torch.Tensor(simplest_data).to(device)

        gpu_result = torch.Tensor(np.zeros([num_pointset, 8])).to(device)
        oper_begin = time.time()
        min_area_rect.torch_launch_min_area_rect(simplest_data_gpu, gpu_result)
        oper_end = time.time()
        gpu_time += oper_end - oper_begin

        cpu_result = np.zeros((num_pointset, 8))
        # CPU operator test
        for j in range(simplest_data.shape[0]):
            # using loop to traverse pointsets input
            oper_begin = time.time()
            result = min_area_rect_cpu(simplest_data[j][None]) 
            oper_end = time.time()
            cpu_result[j] = result.reshape(8)
            cpu_time += oper_end - oper_begin            

        gpu_record.append(gpu_time/num_pointset*1e-9)
        cpu_record.append(cpu_time/num_pointset*1e-9)
        # print(f"GPU version-> avg time: {gpu_time/num_pointset * 1e-9}, num_pointsets: {num_pointset}")
        # print(f"CPU version-> avg time: {cpu_time/num_pointset * 1e-9}, num_pointsets: {num_pointset}")

    num_pointsets = [math.log(i) for i in num_pointsets] 
    times = [math.log(i/j) for i, j in zip(cpu_record, gpu_record)]    
    fig, ax = plt.subplots(layout='constrained')
    plt.plot(num_pointsets, times)
    ax.set_xlabel("ln(num_pointsets)")
    ax.set_ylabel("ln(cpu_cost/gpu_cost)")
    plt.savefig("time_metric.jpg")




