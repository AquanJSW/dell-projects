import multiprocessing as mp
import tqdm
from matchingSAD import *
import cv2
import random

random.seed(5)

N = 10
ldir = "/HDD/tjh/msl/left_nav_pair/"
rdir = "/HDD/tjh/msl/right_nav_pair/"
max_disp = 75
save_path = "./outputs/"

lpaths = list()
rpaths = list()
names = list()
for i in range(N):
    name = random.randint(1, 16068)
    names.append(str(name))
    lpath = ldir + str(name) + ".jpg"
    rpath = rdir + str(name) + ".jpg"
    lpaths.append(lpath)
    rpaths.append(rpath)


def job(q, lpath, rpath):
    Matching(left=lpath, right=rpath, blur=0.5)
    Matching(left=lpath, right=rpath, blur=1)
    Matching(left=lpath, right=rpath, blur=1.5)
    q.put(1)


class Matching(MatchingSAD):
    def __init__(self, left, right, blur):
        super(Matching, self).__init__(max_disparity=max_disp, shift=1, block_size=3, penalty=2, device='cpu',
                                       enable_save=True, save_path=save_path, resize=0.25, left=left, right=right,
                                       blur=blur)

    def save(self):
        try:
            cv2.imwrite(self.save_path + self.name + "_b" + str(self.blur) + '.jpg',
                        self.disparity)
        except IOError:
            print("disparity wrote failed")
        else:
            print("disparity wrote success")


def listener(q):
    bar = tqdm.tqdm(total=N)
    for _ in iter(q.get, None):
        bar.update()


if __name__ == "__main__":
    q = mp.Queue()
    procs = [mp.Process(target=job, args=(q, lpaths[i], rpaths[i])) for i in range(N)]

    pbar = mp.Process(target=listener, args=(q, ))
    pbar.start()

    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    q.put(None)
    pbar.join()
