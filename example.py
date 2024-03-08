import torch
import min_area_rect
import numpy as np
import time

if __name__=="__main__":
    N = 9 # number of points
    # test using the simplest example
    num_pointsets = [1,3,5,10,18,100,1000, 2000, 10000, 50000, 100000]
    device = torch.device('cuda:0')
    for num_pointset in num_pointsets:
        # GPU operator test
        simplest_data = np.random.random([num_pointset ,2*N]) # [1, 18] 
        simplest_data_gpu = torch.Tensor(simplest_data).to(device)

        gpu_result = torch.Tensor(np.zeros([num_pointset, 8])).to(device)
        oper_begin = time.time()
        min_area_rect.torch_launch_min_area_rect(simplest_data_gpu, gpu_result)
        oper_end = time.time()
        gpu_time = oper_end - oper_begin
        print(f"GPU version-> avg time: {gpu_time/num_pointset * 1e-9}, num_pointsets: {num_pointset}")