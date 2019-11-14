# Bag-of-Words
Using Bag of Words to implement Multiple classification
这次任务主要分为几个模块：
  1. 图像特征提取：使用SURF图像特征
  2. 降维阶段：使用Kmeans聚类，形成k个码本（将一副图片降维为k维向量）
  3. 分类训练阶段：使用one vs one 的多分类SVM，对于N个分类需要创建N(N-1)/2个SVM分类器，最后采取投票的方式决定最后分类种类
  
