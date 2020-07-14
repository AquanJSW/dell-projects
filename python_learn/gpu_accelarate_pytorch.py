"""
用pytorch加速矩阵运算
"""

import time
import numpy as np
import torch


# 两个矩阵相乘
def main():
    gpu = torch.device("cuda:0")
    cpu = torch.device("cpu")
    matrix_dim = 5000
    a = torch.arange(start=0, end=matrix_dim * matrix_dim, step=1, dtype=torch.float32).reshape(matrix_dim, -1)
    b = a
    # # cpu start
    # start_time = time.process_time()
    # cpu_product = torch.mm(a, b)
    # end_time = time.process_time()
    # print("cpu time: ", end_time-start_time)
    # print(cpu_product.shape)
    # # cpu end

    # gpu start
    a = a.to(device=gpu)
    b = b.to(device=gpu)
    start_time = time.process_time()
    gpu_product = torch.mm(a, b)
    end_time = time.process_time()
    print("gpu time: ", end_time-start_time)
    print(gpu_product.shape)
    # gpu stop
    gpu_product = gpu_product.to(device=cpu)
    print(gpu_product[1490:, 1490:])


if __name__ == "__main__":
    main()
