# import multiprocessing as mp
import os
import time
import tqdm
import torch.multiprocessing as mp
import torch

device = torch.device('cuda')

def info(title):
    print(title)
    print('module name:\t', __name__)
    print('parent process:\t', os.getppid())
    print('process id:\t\t', os.getpid())


def loop(n, N):
    info('function loop')
    for i in range(N):
        n += torch.tensor(i, dtype=torch.int64, device=device) ** n
    print(n)
    print("Success")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    info('main')
    start = time.time()
    for i in range(20):
        temp = mp.Process(target=loop,
                          args=(torch.tensor(4, dtype=torch.int64, device=device),
                                torch.tensor(4, dtype=torch.int64, device=device)))
        # start = time.time()
        temp.start()
        # temp.join()
        # stop = time.time()
        # print(stop - start)
    stop = time.time()
    print(stop - start)
