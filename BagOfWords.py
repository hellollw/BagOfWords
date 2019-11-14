# -*- coding:utf-8 -*-
# Author； Lu Liwen
# Modified Time:2019.11.14
"""
综合BagOfWords的文件读取，处理，分类
图片预处理阶段：Feature_Detect, 利用SURF特征对图片进行特征提取
降维阶段：Kmeans， 利用Kmeans将衣服图像降维成k个码本的向量
训练阶段：SVM_mul， 对训练集每一个样本（每一个k维向量）进行SVM多分类训练

各个文件夹作用：
data 存放训练过程中生成的数据，包括Kmeans生成的簇中心centroids,各样本分类clusterassement等必要数据
image_training 用于存放训练的图片样本
image_test 用于存放测试用的图片样本
import 用于存放各个模块的代码，如Kmeans,SVM

文件夹存放格式：
不同种类图片分布于不同文件夹,文件夹名称不能重复
图片与存放SURF特征的文件位于同一个文件夹
测试集文件路径和训练集文件路径分开放置

读取文件夹固定格式：
cup, scissors 等以该类物品名称命名，起始时刻应初始化文件种类列表，也就是建立标签的索引file_label_list
每个文件夹内SURF特征数据保存的文件名固定为 desNdarray, 可用loadtxt()函数直接读取，返回一个ndarray数组
每个文件夹内存放特征数量的文件名固定为 keyNumList, 可用loadtxt()函数直接读取，返回一个一维向量（type = ndarray)

遇到错误：
1. savetxt()默认存储的为float型数据，故不能直接读取为整数或者以字符串形式写入
2. labellist为字符串数据，不能使用savetxt()来写入文件中

"""

from numpy import *  # 导入全部函数，不需要在此加上类名限定
import SVM_mul as SVM_mul
import Feature_Detect as Feature_Detect
import Kmeans as Kmeans
import os


# 设置类来管理该函数数据
class BagOfWords:
    def __init__(self, training_path, test_path, store_path, k, C, toler, kTup, maxIter):
        # 路径参数
        self.training_path = training_path
        self.test_path = test_path
        self.store_path = store_path

        # Kmeans参数
        # 输入：
        self.k = k
        # 输出更新：
        self.centroids = []  # ????
        self.clusterassement = []

        # SVM参数
        # 输入：
        self.C = C
        self.toler = toler
        self.kTup = kTup
        self.maxIter = maxIter
        # 输出更新:
        self.SVM_List = []
        self.NumList = []
        self.WholeLabelList = []

        # 训练集参数
        self.m_training = 0
        self.classnum = 0

        # 测试集参数
        self.m_test = 0
        self.wrongrate = 100
        self.CalLabel = []


# 获得文件夹标签索引
# 输入：图片文件夹路径:path
# 输出：文件夹标签索引:file_label_list
def getFileLabelList(path):
    file_label_list = []
    for filename in os.listdir(path):
        if filename not in file_label_list:
            file_label_list.append(filename)
        else:
            raise NameError('文件夹命名错误')
    return file_label_list


# 图片预处理阶段，对所有文件夹内的图片进行SURF特征提取，并写入存放SURF特征的文件：desNdarray和SURF特征数量的文件：KeyNumList
# 输入：存放SURF特征文件夹的路径:path, 文件夹标签索引：file_label_list，存放SURF特征的文件名：默认为desNdarray, 存放特征数量的文件名：默认为KeyNumList
# 输出：将文件写入指定的文件夹
def imagePreprocessing(path, file_label_list, desNdarray_name='desNdarray', KeyNumList_name='KeyNumList'):
    for file_label in file_label_list:
        images_folder = path + file_label + '/'
        Feature_Detect.featureExtract(images_folder, images_folder, desNdarray_name, KeyNumList_name)
    print("图像SURF特征处理完成")


# 降维阶段的数据处理，将所有特征变量整合到一个训练集中进行kmeans训练
# 输入：存放SURF特征文件的文件夹路径:path, 文件夹标间索引: file_label_list, 码本数量:k, 存放SURF特征的文件名：默认为desNdarray
# 输出：簇中心:centroids, 样本分类标签：clusterassement
def KmeansPreprocessing(path, file_label_list, k, desNdarray_name='desNdarray'):
    Kmeans_dataset = []
    # 构造kmeans训练集
    for file_label in file_label_list:
        curdataset = loadtxt(path + file_label + '/' + desNdarray_name)
        Kmeans_dataset.extend(curdataset)
    Kmeans_datamat = mat(Kmeans_dataset)
    # print(shape(Kmeans_datamat))
    #   输入样本矩阵，进行Kmeans聚类
    centroids, clusterassement = Kmeans.biKmeans(Kmeans_datamat, k)
    print("Kmeans聚类完成")
    return centroids, clusterassement


# 分类训练阶段的数据预处理，从SURF特征中降维为k维向量
# 输入： 存放SURF特征的文件夹路径：path, 文件夹标间索引: file_label_list,该文件夹标签值：label, 簇中心：centroids, 存放SURF特征的文件名：默认为desNdarray,
# 存放特征数量的文件名：默认为KeyNumList
# 输出： 样本数据列表:datalist, 样本标签列表:labellist
def classifyPreprocessing(path, file_label_list, centroids, desNdarray_name='desNdarray',
                          keyNumList_name='KeyNumList'):
    # 需要返回的值
    datalist = []
    labellist = []
    K = shape(centroids)[0]  # 获得码本数量
    for file_label in file_label_list:
        # 读取的样本数据
        datasurf = loadtxt(path + file_label + '/' + desNdarray_name)
        datanum = loadtxt(path + file_label + '/' + keyNumList_name)
        # 循环用到的局部变量
        newnum = 0
        for i in range(len(datanum)):
            sample = mat(zeros((1, K)))
            lastnum = newnum
            newnum += int(datanum[i])  # 强制转换为整形数据
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
                add_matrix = mat(zeros((1, K)))
                add_matrix[0, max_index] = 1
                sample += add_matrix  # 构造增量矩阵实现矩阵加法
            datalist.append(sample.A[0])
            labellist.append(file_label)
    datamat = mat(datalist)
    # print(shape(datamat))
    # print(shape(labellist))
    print("分类训练阶段的数据预处理阶段完成，构造完样本图片的k维向量和标签矩阵")
    return datamat, labellist


# 分类训练阶段：使用one vs one方法训练多分类SVM，对于k个种类需要训练k(k-1)/2个SVM分类器
# 输入：训练样本数据：trainingMat, 训练样本标签：trainingLabelList, 约束常数：C，松弛变量:toler, 选择核函数类型：kTup, 最大迭代次数：maxIter
# 输出：k(k-1)/2个SVM分类器参数:SVMList, 种类索引：NumList, 标签列表：WholeLabelList
def classificationTraining(trainingMat, trainingLabel, C, toler, kTup, maxIter):
    SVMList, NumList, WholeLabelList = SVM_mul.TrainMulSVM(trainingMat, trainingLabel, C, toler, kTup, maxIter)
    print("分类训练阶段完成")
    return SVMList, NumList, WholeLabelList


# 训练阶段总和
# 输入：存储数据存储类:BagOfWords
# 输出：更新数据存储类
def bagOfWords_training(bag):
    file_label_list = getFileLabelList(bag.training_path)
    # bag.centroids, bag.clusterassement = KmeansPreprocessing(bag.training_path, file_label_list, bag.k)
    # savetxt(bag.store_path + 'Kmeans/centroids', bag.centroids)
    # savetxt(bag.store_path + 'Kmeans/clusterassement', bag.clusterassement)
    bag.centroids = mat(loadtxt(bag.store_path + 'Kmeans/centroids'))
    datamat, labellist = classifyPreprocessing(bag.training_path, file_label_list, bag.centroids)
    savetxt(store_path + 'SVM/datamat', datamat)
    bag.m_training = len(labellist)
    f = open(bag.store_path + 'SVM/labellist.txt', 'w')
    for label in labellist:
        f.write(label + '\n')
    f.close()
    bag.SVM_List, bag.NumList, bag.WholeLabelList = classificationTraining(datamat, labellist, bag.C, bag.toler,
                                                                           bag.kTup,
                                                                           bag.maxIter)
    bag.classnum = len(bag.WholeLabelList)
    # savetxt(bag.store_path + 'SVM/SVM_List', bag.SVM_List)    #SVMList不能写入，其存放的是SVM的类
    savetxt(bag.store_path + 'SVM/NumList', bag.NumList)
    f = open(bag.store_path + 'SVM/WholeLabelList.txt', 'w')
    for label in bag.WholeLabelList:
        f.write(label + '\n')
    f.close()

    f = open(bag.store_path + 'training/readme.txt', 'w')
    f.write('训练集数量：%d' % bag.m_training + '\n')
    f.write('训练集种类个数：%d' % bag.classnum + '\n')
    f.close()


# 测试阶段：输入测试样本，预测测试样本的类型
# 输入：存储数据存储类:BagOfWords
# 输出：更新数据存储类，测试集测试标签：callabel, 测试集误差：wrongrate
def bagOfWords_test(bag):
    test_file_label_list = getFileLabelList(bag.test_path)
    imagePreprocessing(path=bag.test_path, file_label_list=test_file_label_list)
    testmat, labeltest = classifyPreprocessing(bag.test_path, test_file_label_list, bag.centroids)
    bag.CalLabel, bag.WrongRate = SVM_mul.Test(bag.SVM_List, testmat, labeltest, bag.NumList, bag.WholeLabelList)
    bag.m_test = len(labeltest)
    print("测试集测试完成，错误率为:%d" % bag.WrongRate)
    f = open(bag.store_path + 'test/readme.txt', 'w')
    f.write('测试集个数:%d' % bag.m_test + '\n')
    f.write('测试集误差：%d' % bag.WrongRate + '\n')
    f.close()
    f = open(bag.store_path + 'test/CalLabel.txt', 'w')
    for Cal in bag.CalLabel:
        f.write(Cal[0]+' '+str(Cal[1])+'\n')
    f.close()


if __name__ == '__main__':
    training_path = './image_training/'  # 定义图片文件夹路径
    test_path = './image_test/'
    store_path = './data/'
    k = 50  # 定义码本数量
    C = 10  # 定义约束常数
    toler = 0.0001  # 定义松弛变量
    kTup = ('lin', 1.3)  # 定义核函数
    maxIter = 500  # 定义最大迭代数量
    bag = BagOfWords(training_path, test_path, store_path, k, C, toler, kTup, maxIter)
    bagOfWords_training(bag)
    bagOfWords_test(bag)

    # file_label_list = getFileLabelList(bag.training_path)

    # 检验图像处理阶段 已通过
    # imagePreprocessing(image_path,file_label_list)

    # 检验Kmeans聚类阶段 已通过
    # centroids, clusterassement = KmeansPreprocessing(image_path, file_label_list, k)
    # savetxt('./test/centroids',centroids)
    # savetxt('./test/clusterassement',clusterassement)

    # 检验分类训练阶段的数据预处理 已通过
    # centroids = loadtxt('./test/centroids')
    # datamat, labellist = classifyPreprocessing(bag.training_path, file_label_list, centroids)
    # savetxt('./test/datamat', datamat)
    # f = open('./test/labellist.txt', 'w')
    # for label in labellist:
    #     f.write(label + '\n')
    #
    # # 检验分类训练阶段 已通过
    # classificationTraining(datamat, labellist, C, toler, kTup, maxIter)
