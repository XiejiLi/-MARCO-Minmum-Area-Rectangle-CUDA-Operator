'''
Usage
min_area_rect_torch_min_area_rect(Tensor pointsets, int N, Tensor result)
    Params:
        pointsets: -> Tensor [N, 18], N is the number of points
        N: -> int number of points
    Return:
        result -> Tensor [N, 8]
'''
import torch
import min_area_rect
import numpy as np
import matplotlib.pyplot as plt

def visualize(pointsets, result):
    plt.plot(pointsets[::2], pointsets[1::2], 'ro')
    result = np.insert(result, 0, result[-2:], axis=0)
    plt.plot(result[::2], result[1::2], 'b')


if __name__=="__main__":
    N = 9 # number of points
    # test using the simplest example
    num = 4
    device = torch.device('cuda:0')
    simplest_data = np.random.random([num ,2*N]) # [1, 18] 
    result = torch.Tensor(np.zeros([num, 8])).to(device)

    simplest_data_gpu = torch.Tensor(simplest_data).to(device)
    min_area_rect.torch_launch_min_area_rect(simplest_data_gpu, result)

    for i in range(num):
        plt.subplot(2,int(num/2), i+1)
        visualize(simplest_data[i], result[i].cpu().numpy())
    plt.savefig('test_result.png')