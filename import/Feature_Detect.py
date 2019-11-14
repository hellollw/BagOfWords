# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-13

"""
使用SURF特征提取提取图片特征

改进代码：
1.featureExtract当写入文件失败时return false,抛出写入文件错误

修改问题：
1. 读取文件中图片失败：
    图片路径包含中文
2. 写入SURF描述矩阵失败：
    写入的SURF特征描述矩阵的desNdarray,KeyNumList名字（str格式）与变量名重合，将变量desNdarray,KeyNumList修改为desNdarray_name,KeyNumList_name

优化思考：
1.进行轮廓边缘检测再进行轮廓的特征提取？（因为物品上的动画图案可能会使训练过拟合，需要更加注重于物品的轮廓特征！）——先进行Canny边缘检测？ 有什么轮廓特征提取的方法？
"""

import cv2 as cv
import numpy as np
import os


# 对一个文件夹内的图片都进行SURF特征提取,同时将对图片的SURF特征提取值直接以数组形式保存在文件中，可直接利用np.loadtxt()读取数据，返回的为ndarray(float组成的数据！）
# 输入：图片文件夹路径：images_folder， 写出文件保存路径：Output_Path, 存放SURF特征的文件名：默认为desNdarray, 存放特征数量的文件名：默认为KeyNumList
# 输出：存储图片的特征描述矩阵文件：desNdarray, 每一副图片的特征点个数列表：KeyNumList
def featureExtract(images_folder, OutputPath, desNdarray_name='desNdarray', KeyNumList_name='KeyNumList'):
    KeyNumList = []
    desNdarray = []
    for filename in os.listdir(images_folder):  # 打印目录的所有文件（遍历图片目录的每一幅图片）
        if '.jpg' in filename:
            filePath = images_folder + filename
        else:
            continue
        img = cv.imread(filename=filePath)
        blurred = cv.GaussianBlur(img, (3, 3), 0)   #高斯模糊降噪
        gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)  # 转换成灰度图像

        # #先使用边缘检测再进行SURF特征提取？
        # edge_output = cv.Canny(gray,50,150)
        # cv.imshow('canny',edge_output)
        # cv.waitKey(0)

        detector = cv.xfeatures2d.SURF_create(1000)  # 创建SURF
        kps, des = detector.detectAndCompute(gray, None)  # 进行SURF特征提取，返回图片特征的关键点和描述符

        # img2 = img.copy()
        # img2 = cv.drawKeypoints(image=img, outImage=img2, keypoints=kps, color=(255, 0, 0))  # 绘制关键点
        # print(type(img2))
        # cv.imshow('kps',img2)
        # cv.waitKey(0)

        print(np.shape(des))
        desNdarray.extend(des)  # 整合描述符(存储为2维list）
        KeyNumList.append(len(kps))  # 整合每幅图片包含的关键点个数
    try:
        print('写入的描述矩阵维度为:', np.shape(np.array(desNdarray)))  # 每次读取完一个文件夹提示一个写入维度指明信息
        np.savetxt(OutputPath + desNdarray_name, desNdarray)
        np.savetxt(OutputPath + KeyNumList_name, KeyNumList)
    except:
        print('写入图片文件夹' + OutputPath + '失败')
        return False
    else:
        print('写入图片文件夹' + OutputPath + '成功')
        return desNdarray, KeyNumList


if __name__ == '__main__':
    images_folder = './image/Transformers/'  # 尝试给出图片文件夹
    # OutputPath = './imageInfo/'  # 给出保存文件路径
    featureExtract(images_folder=images_folder, OutputPath=images_folder)
    # 尝试读出数据
    # desNdarray = np.loadtxt(images_folder + 'desNdarray')
    # print(np.shape(desNdarray))
    # KeyNumList = np.loadtxt(images_folder + 'KeyNumList')
    # print(np.shape(KeyNumList))
