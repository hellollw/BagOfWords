# Bag-of-Words
Using Bag of Words to implement Multiple classification
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
