"""
概览:
v1目标是利用卷积优化匹配
v1只适用于SAD代价计算

结果：
实现在gpu上卷积，实现在gpu上用multiprocessing多线程代价比较，但效率仍然不高

出发点:
未极线校准的左右导航相机图像存在纵向小幅度相差（-10px ~ 10px）,故考虑：
1   匹配范围：对于左图中的待匹配像素p，在右图中的长为最大视差D(暂定250)，宽为10+10=20的矩形像素内匹配像素
2   匹配框：由于图像中相当一部分缺少纹理，应该增大匹配框的大小（暂定7*7）,注意padding
3   代价：暂定SAD
4   原则：代价最小的像素的横坐标与待匹配像素横坐标之差最为待匹配像素的视差
用gpu匹配时，考虑先将两幅padding后的图像用对应匹配框大小的全1卷积核卷积，得到两幅尺寸与padding前图像相同的图像
这样，卷积后的两幅图像对应位置的像素值就是原图像以该位置为中心的匹配框内像素值之和

TODO:
1   一张图像，越远离中心，畸变越大
    应该找到图像的畸变规律，实现图像位置敏感的匹配范围以优化精度与速度
2   尝试更优的匹配代价公式
3   增加降采样的参数，最后上采样输出，加快处理速度

IDEA:
1   已经可以利用卷积计算代价，对于寻找最小代价的任务，可以像matching_cpu中一样手动分配匹配任务，并行cpu加速，
    （如果直接用torch进行for循环不能加速）也可以考虑手动给cuda核心分配任务加速匹配
"""

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as multiprocessing
# import multiprocessing
import torchvision.transforms as transforms
import PIL.Image as Image
import cv2

device = torch.device("cuda")

# 用torch优化过的conv2d函数实现快速卷积，并不需要学习weight，这里给定weight实现单次卷积并输出
# 但这里的全1卷积核所实现的核内求和只适用于SAD代价计算
class Net(nn.Module):
    def __init__(self, kernel_size):
        super(Net, self).__init__()
        kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float16, requires_grad=False)
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(device=device)
        # nn.Parameter也可以接受cuda类型的张量
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, img):
        img = F.conv2d(img.unsqueeze(0).unsqueeze(0), self.weight)
        return img.squeeze()


class Matching(object):
    def __init__(self, left_img="/home/tjh/dataset/msl/left_nav_pair/10268.jpg",
                 right_img="/home/tjh/dataset/msl/right_nav_pair/10268.jpg", max_disparity=30,
                 max_vertical_shift=1, min_vertical_shift=1, block_size=5,
                 min_threshold=10, max_threshold=1250):
        self.left_img = cv2.imread(left_img, cv2.IMREAD_GRAYSCALE)
        self.right_img = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE)
        self._img_resize(256)
        self.img_height = self.left_img.shape[0]
        self.max_disparity = max_disparity
        self.vertical_shift = [-min_vertical_shift, max_vertical_shift]
        self.block_size = block_size
        self._step = self.block_size // 2
        self.disparity = np.zeros_like(self.left_img, dtype="uint8")

        # 判断block_size是否是奇数
        if not (self.block_size % 2):
            raise ValueError("block size must be an odd number")

        # 用cv2的padding而不是F.conv2d参数中的padding
        # cv2.BORDER_WRAP可以实现对边映射padding，而非简单地恒定值，
        # 可以一定程度上减少误匹配的几率
        self.right_img_padded = cv2.copyMakeBorder(self.right_img, self._step, self._step,
                                                   self._step, self._step, cv2.BORDER_WRAP)
        self.left_img_padded = cv2.copyMakeBorder(self.left_img, self._step, self._step,
                                                  self._step, self._step, cv2.BORDER_WRAP)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def main(self):
        # cores = multiprocessing.cpu_count()
        cores = 1
        tasks = self._task_separate(cores)
        pool = multiprocessing.Pool(processes=cores)
        self.left_img_padded = torch.from_numpy(self.left_img_padded).to(device=device, dtype=torch.float16)
        self.right_img_padded = torch.from_numpy(self.right_img_padded).to(device=device, dtype=torch.float16)

        # 经过卷积后就可以直接对比像素值了,因为此时的像素值已经是原图像中的block中像素值之和了
        net = Net(self.block_size)
        self.left_img_conved = net(self.left_img_padded)
        self.right_img_conved = net(self.right_img_padded)

        disparities = pool.map(self.matching, tasks)
        disparity = []
        for disp in disparities:
            disparity += disp
        self.disparity = np.array(disparity).reshape(self.left_img.shape[0], -1)

        try:
            cv2.imwrite("./outputs/disparity_b{}t{}-{}.jpg".format(self.block_size, self.min_threshold, self.max_threshold),
                        self.disparity)
        except IOError:
            print("Failed written disparity map~!")
        else:
            print("Successfully written disparity map~!")

    def _img_resize(self, img_size):
        self.left_img = cv2.resize(self.left_img, (img_size, img_size))
        self.left_img = cv2.resize(self.right_img, (img_size, img_size))

    def _task_separate(self, cores):
        # 仅适用于方形图像
        # 先将像素坐标存入一维数组
        left_img_idx = []
        for row in range(self.img_height):
            for column in range(self.img_height):
                left_img_idx.append((row, column))
        # 然后按cpu核心数分配子任务块
        pixel_num = len(left_img_idx)
        subtask_size = pixel_num // cores
        tasks = []
        for split in range(0, pixel_num, subtask_size):
            tasks.append(left_img_idx[split: split + subtask_size])
        return tasks

    def matching(self, left_img_idxes):
        print("threshold: [{}, {}]".format(self.min_threshold, self.max_threshold), "block size:", self.block_size)
        # 直接创建一个全255的数组，使接下来的if语句不用重复赋值255
        disparity = torch.zeros(len(left_img_idxes), device=device, dtype=torch.uint8) + 255
        disparity = disparity.to(device="cpu").numpy().tolist()
        for idx, left_img_idx in enumerate(tqdm(left_img_idxes)):
            search_pixels = self._search_pixels(left_img_idx)
            costs = []
            for right_img_idx in search_pixels:
                costs.append((self.left_img_conved[left_img_idx[0], left_img_idx[1]] -
                              self.right_img_conved[right_img_idx[0], right_img_idx[1]]).abs().item())
            min_cost = min(costs)
            # if (min_cost > self.max_threshold or
            #         min_cost <= self.min_threshold):
            #     disparity.append(255)
            # else:
            #     matched_column = search_pixels[costs.index(min_cost)][1]
            #     disparity.append(left_img_idx[1] - matched_column)
            if (min_cost < self.max_threshold) & (min_cost > self.min_threshold):
                matched_column = search_pixels[costs.index(min_cost)][1]
                disparity[idx] = left_img_idx[1] - matched_column
        # print("Current process: ", multiprocessing.current_process(),
            #       " Pixel {} finished matching~!".format(left_img_idx))
        return disparity

    def show(self):
        plt.subplot(131), plt.imshow(self.left_img), plt.title("left")
        plt.subplot(132), plt.imshow(self.disparity), plt.title("disparity")
        plt.subplot(133), plt.imshow(self.right_img), plt.title("right")
        plt.show()

    # 返回待匹配像素在右图中的可能匹配像素
    # 返回形式：list of lists: [[1,1], [2,2]]
    def _search_pixels(self, left_img_px):
        # 定位匹配范围的左上角和右下角坐标
        left_upper_corner = [left_img_px[0] + self.vertical_shift[0],
                             left_img_px[1] - self.max_disparity]
        right_bottom_corner = [left_img_px[0] + self.vertical_shift[1],
                               left_img_px[1]]
        if left_upper_corner[0] < 0:
            left_upper_corner[0] = 0

        if left_upper_corner[1] < 0:
            left_upper_corner[1] = 0

        if right_bottom_corner[0] >= self.img_height:
            right_bottom_corner[0] = self.img_height

        search_pixels = []
        for row in range(left_upper_corner[0], right_bottom_corner[0] + 1):
            for column in range(left_upper_corner[1], right_bottom_corner[1] + 1):
                search_pixels.append([row, column])
        return search_pixels


if __name__ == "__main__":
    m = Matching()
    m.main()