# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    left = cv2.imread("/home/tjh/projects/py/disparity/left/NLB_606739103EDR_F0750720NCAM00385M1.png", 0)
    right = cv2.imread("/home/tjh/projects/py/disparity/right/NRB_606739103EDR_F0750720NCAM00385M1.png", 0)
    print('type of image: ', type(left))
    print('size of image: ', left.shape)
    # 用cv绘图
    # cv2.namedWindow('on mars', cv2.WINDOW_NORMAL)
    # cv2.imshow('on mars', left)
    # k = cv2.waitKey(0)
    # print("key's ascii: ", k)
    # cv2.destroyWindow('on mars')
    # matplotlib 画图
    # plt.imshow(left, cmap='gray')
    # plt.show()

    # 用matplotlib绘图
    # 划线
    line_num = 10
    concatenated = np.hstack([left, right])
    for n in range(line_num):
        step = concatenated.shape[0] // (line_num + 1)
        start = (0, step * (n+1))
        stop = (concatenated.shape[1], step * (n+1))
        # (img, start_point, stop_point, color, thickness, line_type)
        cv2.line(concatenated, start, stop, 255, 2, lineType=cv2.LINE_AA)
    cv2.line(concatenated, (concatenated.shape[1] // 2, 0),
             (concatenated.shape[1] // 2, concatenated.shape[0]) )
    plt.imshow(concatenated, 'gray')
    plt.show()
    # # (..., 左下角，右上角，...), 画矩形
    # cv2.rectangle(left, (380, 570), (790, 300), 0, 3, lineType=cv2.LINE_AA)
    # # 写字
    # font = cv2.FONT_ITALIC
    # cv2.putText(left, 'On Mars', (512, 100), font, 4, 0, 5, lineType=cv2.LINE_AA)

    # 鼠标事件
#    events = [i for i in dir(cv2) if 'EVENT' in i]
#    print()
#    print('Mouse Events:')
#    for i in events:
#        print(i)
#    cv2.namedWindow('left', cv2.WINDOW_NORMAL)
#    cv2.setMouseCallback('left', draw_circle, left)
#    while 1:
#        cv2.imshow('left', left)
#        if cv2.waitKey() == 's':
#            break
#    cv2.destroyWindow()

    # 切片
    # subimg = left[0:100, 100:200]
    # cv2.namedWindow('subimg')
    # cv2.imshow('subimg', subimg)

    # padding
    # wrapped = cv2.copyMakeBorder(left, 10, 10, 10, 10, cv2.BORDER_DEFAULT)
    # cv2.namedWindow('wrapped')
    # cv2.imshow('wrapped', wrapped)

    # alpha channel
    # a = 0.5
    # mixed = a * left + (1-a) * right
    # plt.subplot(131), plt.imshow(left, 'gray'), plt.title('left')
    # plt.subplot(132), plt.imshow(mixed, 'gray'), plt.title('mixed')
    # plt.subplot(133), plt.imshow(right, 'gray'), plt.title('right')
    # plt.show()

    # bit-wise compute
    # T, left_mask = cv2.threshold(left, 90, 255, cv2.THRESH_BINARY)
    # print('left_mask: \n', left_mask)
    # left_mask_inv = cv2.bitwise_not(left_mask)
    # print('left_mask_inv: \n', left_mask_inv)
    # # left_masked = cv2.bitwise_and(left, left, mask=left_mask_inv)
    # left_masked = cv2.bitwise_and(left, left_mask_inv)
    # print('left_masked:\n', left_masked)
    # right_masked = cv2.bitwise_and(right, left_mask)
    # merge = cv2.add(left_masked, right_masked)
    # cv2.namedWindow('appended')
    # cv2.imshow('appended', merge)

    # resize
    # mini_left = cv2.resize(left, (60, 60), interpolation=cv2.INTER_AREA)
    # mini_right = cv2.resize(right, (60, 60), interpolation=cv2.INTER_AREA)
    # cv2.namedWindow('mini_left')
    # cv2.imshow('mini_left', mini_left)


    # disparity

    # stereo = cv2.StereoSGBM_create(numDisparities=288, blockSize=7, P1=4, P2=8)

    # disparity = stereo.compute(left, right)
    # plt.subplot(131), plt.imshow(left, 'gray'), plt.title('left')
    # plt.subplot(132), plt.imshow(disparity, "gray"), plt.title('disparity')
    # plt.subplot(133), plt.imshow(right, 'gray'), plt.title('right')
    # plt.show()

    k = cv2.waitKey(0)
    if k == 's':
        cv2.destroyWindow()


def draw_circle(event, x, y, flags, param):
    img = param
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (512, 512), 100, 255, 2, lineType=cv2.LINE_AA)


if __name__ == '__main__':
    main()
