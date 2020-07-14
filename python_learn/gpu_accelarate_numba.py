"""
学习如何用gpu加速python
结论，numba限制较多，有些numpy运算法则不能使用
"""

from numba import cuda
import time
import numpy as np


# 两个矩阵相乘
def main():
    matrix_dim = 1600
    a = np.arange(matrix_dim*matrix_dim, dtype=np.int32).reshape((matrix_dim, matrix_dim))
    b = a
    # cpu start
    start_time = time.process_time()
    cpu_product = np.dot(a, b)
    end_time = time.process_time()
    print("cpu time: ", end_time-start_time)
    # cpu end

    # gpu+cpu start
    start_time = time.process_time()

    # 将矩阵相乘先作 vector(1*n) * vector(n*1) 得到一个list (其元素累加和即得最终矩阵积)
    intermediate_products = np.zeros(matrix_dim, dtype=np.int32)
    block_dim = 192     # == cuda cores
    grid_dim = int(np.ceil(matrix_dim / block_dim))
    gpu_matrix_product[grid_dim, block_dim](a, b, intermediate_products, matrix_dim)

    # list元素累加和
    gpu_product = np.zeros((matrix_dim, matrix_dim), dtype=np.int32)
    threads = matrix_dim * matrix_dim
    block_dim = 192
    grid_dim = int(np.ceil(threads / block_dim))
    gpu_matrix_add(intermediate_products, gpu_product, threads, matrix_dim)
    # gpu+cpu end


@cuda.jit()
def gpu_matrix_add(matrix_list, result, threads, matrix_dim):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < threads:
        axis0 = threads // matrix_dim
        axis1 = threads // matrix_dim
        for matrix in matrix_list:
            threads[axis0, axis1] += matrix[axis0, axis1]


@cuda.jit()
def gpu_matrix_product(a, b, result, threads):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < threads:
        result[idx] = cuda.np.dot(a[:, idx].reshape(threads,-1), b[idx, :].reshape(-1, threads))


if __name__ == "__main__":
    main()