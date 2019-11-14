# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-13

"""
使用SURF特征提取提取图片特征

"""

import cv2 as cv
import numpy as np
import sys
import os


# 对一个文件夹内的图片都进行SURF特征提取,同时将对图片的SURF特征提取值直接以数组形式保存在文件中，可直接利用np.loadtxt()读取数据，返回的为ndarray(float组成的数据！）
# 输入：图片文件夹路径：images_folder， 写出文件保存路径：Output_Path, 存放SURF特征的文件名：默认为desNdarray, 存放特征数量的文件名：默认为KeyNumList
# 输出：存储图片的特征描述矩阵文件：desNdarray, 每一副图片的特征点个数列表：KeyNumList
def featureExtract(images_folder, OutputPath, desNdarray='desNdarray', KeyNumList='KeyNumList'):
    KeyNumList = []
    desNdarray = []
    for filename in os.listdir(images_folder):  # 打印目录的所有文件（遍历图片目录的每一幅图片）
        if '.jpg' in filename:
            filePath = images_folder + filename
        else:
            continue
        img = cv.imread(filename=filePath)
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # 转换成灰度图像
        detector = cv.xfeatures2d.SURF_create(2000)  # 创建SURF
        kps, des = detector.detectAndCompute(gray, None)  # 进行SURF特征提取，返回图片特征的关键点和描述符
        img = cv.drawKeypoints(image=img, outImage=img, keypoints=kps,
                               flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(255, 0, 0))  # 绘制关键点
        print(type(des))
        print(np.shape(des))
        desNdarray.extend(des)  # 整合描述符(存储为2维list）
        KeyNumList.append(len(kps))  # 整合每幅图片包含的关键点个数
    try:
        print(np.shape(np.array(desNdarray)))
        np.savetxt(OutputPath + desNdarray, desNdarray)
        np.savetxt(OutputPath + KeyNumList, KeyNumList)
    except:
        print('写入图片文件夹' + OutputPath + '失败')
    else:
        print('写入图片文件夹' + OutputPath + '成功')
    return desNdarray, KeyNumList


if __name__ == '__main__':
    images_folder = './test/'  # 尝试给出图片文件夹
    OutputPath = './imageInfo/'  # 给出保存文件路径
    featureExtract(images_folder=images_folder, OutputPath=OutputPath)
    # 尝试读出数据
    desNdarray = np.loadtxt(OutputPath + 'desNdarray')
    print(np.shape(desNdarray))
    print(type(desNdarray[1, 1]))
    KeyNumList = np.loadtxt(OutputPath + 'KeyNumList')
    print(np.shape(KeyNumList))
    print(KeyNumList)
