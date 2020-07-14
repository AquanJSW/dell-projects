"""
v0仅仅在cost函数作了tensor化
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
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import cv2

device = torch.device("cuda")

class Matching(object):
    def __init__(self, left_img="/home/tjh/dataset/msl/left_nav_pair/10268.jpg",
                 right_img="/home/tjh/dataset/msl/right_nav_pair/10268.jpg", max_disparity=150,
                 max_vertical_shift=2, min_vertical_shift=2, block_size=7):

        self.left_img = cv2.imread(left_img, cv2.IMREAD_GRAYSCALE)
        self.right_img = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE)
        self.max_disparity = max_disparity
        self.vertical_shift = [-min_vertical_shift, max_vertical_shift]
        self.block_size = block_size
        self._step = self.block_size // 2
        self.disparity = np.zeros_like(self.left_img, dtype="uint8")

    def start_matching(self):
        self._padding()
        for row in range(self._step, self.left_img_padded.shape[0] - self._step):
            for column in range(self._step, self.left_img_padded.shape[1] - self._step):
                left_img_px = [row, column]
                search_pixels = self._search_pixels(left_img_px)
                cost = []
                for right_img_px in search_pixels:
                    cost.append(self._cost(left_img_px, right_img_px))
                matched_column = search_pixels[cost.index(min(cost))][1]
                self.disparity[row - self._step, column - self._step] = column - matched_column
                print("Pixel [", row - self._step, column - self._step, "] finished matching!")

        try:
            cv2.imwrite("./disparity.jpg", self.disparity)
        except IOError:
            print("Failed written disparity map~!")
        else:
            print("Successfully written disparity map~!")

    def show(self):
        plt.subplot(131), plt.imshow(self.left_img), plt.title("left")
        plt.subplot(132), plt.imshow(self.disparity), plt.title("disparity")
        plt.subplot(133), plt.imshow(self.right_img), plt.title("right")
        plt.show()

    # 注意padding后图像的尺寸也变了，匹配时不要使padding像素成为待匹配像素
    def _padding(self):
        if not (self.block_size % 2):
            raise ValueError("block size must be an odd number")

        self.right_img_padded = cv2.copyMakeBorder(self.right_img, self._step, self._step,
                                                   self._step, self._step, cv2.BORDER_WRAP)
        self.left_img_padded = cv2.copyMakeBorder(self.left_img, self._step, self._step,
                                                  self._step, self._step, cv2.BORDER_WRAP)

    def _cost(self, left_img_centre_px, right_img_centre_px):
        left_img_block_index = np.ix_(list(range(left_img_centre_px[0] - self._step,
                                                 left_img_centre_px[0] + self._step + 1)),
                                      list(range(left_img_centre_px[1] - self._step,
                                                 left_img_centre_px[1] + self._step + 1)))
        right_img_block_index = np.ix_(list(range(right_img_centre_px[0] - self._step,
                                                  right_img_centre_px[0] + self._step + 1)),
                                       list(range(right_img_centre_px[1] - self._step,
                                                  right_img_centre_px[1] + self._step + 1)))
        left_img_block = self.left_img_padded[left_img_block_index]
        right_img_block = self.right_img_padded[right_img_block_index]

        # SAD(Sum of Absolute Difference) cost
        left_img_block = torch.from_numpy(left_img_block)
        left_img_block = torch.as_tensor(left_img_block, dtype=torch.float16)
        right_img_block = torch.from_numpy(right_img_block)
        right_img_block = torch.as_tensor(right_img_block, dtype=torch.float16)
        left_img_block = left_img_block.to(device=device)
        right_img_block = right_img_block.to(device=device)
        cost = (left_img_block - right_img_block).abs().sum().item()
        # cost = torch.dot(torch.dot(ones_vector,
        #                            (left_img_block - right_img_block).abs()),
        #               ones_vector.T).squeeze().item()
        return cost

    # 返回待匹配像素在右图中的可能匹配像素
    # 返回形式：list of lists: [[1,1], [2,2]]
    def _search_pixels(self, left_img_px):
        # 定位匹配范围的左上角和右下角坐标
        left_upper_corner = [left_img_px[0] + self.vertical_shift[0],
                             left_img_px[1] - self.max_disparity]
        right_bottom_corner = [left_img_px[0] + self.vertical_shift[1],
                               left_img_px[1]]
        if left_upper_corner[0] < self._step:
            left_upper_corner[0] = self._step

        if left_upper_corner[1] < self._step:
            left_upper_corner[1] = self._step

        if right_bottom_corner[0] > (self.left_img_padded.shape[0] - self._step - 1):
            right_bottom_corner[0] = self.left_img_padded.shape[0] - self._step - 1

        search_pixels = []
        for row in range(left_upper_corner[0], right_bottom_corner[0] + 1):
            for column in range(left_upper_corner[1], right_bottom_corner[1] + 1):
                search_pixels.append([row, column])
        return search_pixels


if __name__ == "__main__":
    m = Matching()
    m.start_matching()
