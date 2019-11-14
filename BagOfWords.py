# -*- coding:utf-8 -*-
# Author； Lu Liwen
# Modified Time:2019.11.14
"""
综合BagOfWords的文件读取，处理，分类
图片预处理阶段：Feature_Detect, 利用SURF特征对图片进行特征提取
降维阶段：Kmeans， 利用Kmeans将衣服图像降维成k个码本的向量
训练阶段：SVM_mul， 对训练集每一个样本（每一个k维向量）进行SVM多分类训练

读取文件夹固定格式：
cup, scissors 等以该类物品名称命名，起始时刻应初始化文件种类列表，也就是建立标签的索引file_label_list
每个文件夹内SURF特征数据保存的文件名固定为 desNdarray, 可用loadtxt()函数直接读取，返回一个ndarray数组
每个文件夹内存放特征数量的文件名固定为 keyNumList, 可用loadtxt()函数直接读取，返回一个一维向量（type = ndarray)

"""

from numpy import *  # 导入全部函数，不需要在此加上类名限定
import SVM_mul as SVM_mul
import Feature_Detect as Feature_Detect
import Kmeans as Kmeans


# 图片预处理阶段，对所有文件夹内的图片进行SURF特征提取，并写入存放SURF特征的文件：desNdarray和SURF特征数量的文件：KeyNumList
# 输入：存放SURF特征文件夹的路径:path, 文件夹标签索引：file_label_list，存放SURF特征的文件名：默认为desNdarray, 存放特征数量的文件名：默认为KeyNumList
# 输出：将文件写入指定的文件夹
def imagePreprocessing(path, file_label_list, desNdarray='desNdarray', KeyNumList='KeyNumList'):
    for file_label in file_label_list:
        images_folder = path + file_label
        Feature_Detect.featureExtract(images_folder, images_folder, desNdarray, KeyNumList)
    print("图像SURF特征处理完成")


# 降维阶段的数据处理，将所有特征变量整合到一个训练集中进行kmeans训练
# 输入：存放SURF特征文件的文件夹路径:path, 文件夹标间索引: file_label_list, 码本数量:k, 存放SURF特征的文件名：默认为desNdarray
# 输出：簇中心:centroids, 样本分类标签：clusterassement
def KmeansPreprocessing(path, file_label_list, k, desNdarray='desNdarray'):
    Kmeans_dataset = []
    # 构造kmeans训练集
    for file_label in file_label_list:
        curdataset = loadtxt(path + file_label + desNdarray)
        Kmeans_dataset.extend(curdataset)
    Kmeans_datamat = mat(Kmeans_dataset)
    #   输入样本矩阵，进行Kmeans聚类
    centroids, clusterassement = Kmeans.biKmeans(Kmeans_datamat, k)
    return centroids, clusterassement


# 分类训练阶段的数据预处理，从SURF特征中降维为k维向量
# 输入： 存放SURF特征的文件夹路径：path, 文件夹标间索引: file_label_list,该文件夹标签值：label， 码本数量：k, 簇中心：centroids, 存放SURF特征的文件名：默认为desNdarray,
# 存放特征数量的文件名：默认为KeyNumList
# 输出： 样本数据列表:datalist, 样本标签列表:labellist
def classifyPreprocessing(path, file_label_list, k, centroids, desNdarray='desNdarray', keyNumList='KeyNumList'):
    # 需要返回的值
    datalist = []
    labellist = []
    for file_label in file_label_list:
        # 读取的样本数据
        datasurf = loadtxt(path + file_label + desNdarray)
        datanum = loadtxt(path + file_label + keyNumList)
        # 循环用到的局部变量
        newnum = 0
        for i in range(len(datanum)):
            sample = mat(zeros((1, k)))
            lastnum = newnum
            newnum += datanum[i]
            for num in range(lastnum, newnum):  # 从上一个特征向量数取至下一个特征向量数
                surf = datasurf[num, :]  # 取出一个surf特征
                max_dist = inf
                max_index = 0
                for centroid in range(len(centroids)):  # 对每一个码本进行遍历
                    dist = Kmeans.distEclud(centroids[centroid, :], surf)  # 使用欧氏距离计算两向量之间距离
                    if dist < max_dist:
                        max_dist = dist
                        max_index = centroid
                    else:
                        continue
                add_matrix = mat(zeros((1, k)))
                add_matrix[1, max_index] = 1
                sample += add_matrix  # 构造增量矩阵实现矩阵加法
            datalist.append(sample)
            labellist.append(file_label)
    datamat = mat(datalist)
    return datamat, labellist

# 分类训练阶段：使用one vs one方法训练多分类SVM，对于k个种类需要训练k(k-1)/2个SVM分类器
# 输入：训练样本数据：trainingList, 训练样本标签：trainingLabelList, 约束常数：C，松弛变量:toler, 选择核函数类型：kTup, 最大迭代次数：maxIter
# 输出：k(k-1)/2个SVM分类器参数:SVMList, 种类索引：NumList, 标签列表：WholeLabelList
def classificationTraining(trainingMat, trainingLabel, C, toler, kTup, maxIter):
    return SVM_mul.TrainMulSVM(trainingMat, trainingLabel, C, toler, kTup, maxIter)
