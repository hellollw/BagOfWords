from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv



# dataset = [[1, 2, 3], [2, 2, 6], [7, 8, 9]]
# data = mat([[1,2,3],[4,5,6]])
#
# dataset[1] = data[1,:].A[0]
# print(dataset)

# data_mat = mat(dataset)
# data_mat1 = dataset.copy()
# data_mat2 = dataset
# dataset = dataset.append([1,4,5])
# print(data_mat1)
# print(data_mat2)

# data.append([1,2,3])
# print(data_mat)
# notzero = nonzero(data_mat[:,1]==2)[0]
# print(data_mat)
# print(type(notzero))
# datamat = mat(dataset)
# data_max = max(datamat[1, :])
# print(data_max)
# print(5 * random.rand(5, 1))  # 产生5个0~1之间的随机数

# colmean = mean(datamat, axis=0)
# print(colmean.tolist()[0])
# dataset[1] = colmean  # 只有矩阵形式可以用:来取值
# print(dataset)

# 当前工作路径是指当前的py文件下文件夹的路径
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split('\t')  # strip()为去除首尾字符，split为分割出字符串('\t'是键盘上的tab键）
        if len(curline) != 1:
            filLine = list(map(float, curline))  # 含有空的字符串
            dataMat.append(filLine)

    # print(mat(dataMat))
    return dataMat


# 先只测试能够绘画代码
def plotdata(dataSet):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ['+', 'o', '*', 'x', 'd', '.', 'd', '^']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    data_x = dataSet[:, 0].flatten().A[0]
    data_y = dataSet[:, 1].flatten().A[0]
    ax.scatter(data_x, data_y, color='r', marker='+')  # 一次画多个点

    # rect = [0.1,0.1,1.2,1.2]
    # ax = fig.add_axes(rect,label='ax')
    plt.show()

# 测试random函数
def try_class(os):
    os.m = 1

def try_class2():
    return 0

class data():
    def __init__(self):
        self.datam = []
        self.m = 0

if __name__ == '__main__':
    # img = cv.imread(filename='./test/55.bmp')
    # gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # 转换成灰度图像
    # cv.imshow('gray',gray)
    # cv.waitKey(0)
    # i=0
    # i+=try_class2()
    # print('i: %d dsadsa'%i)
    # os = data
    # try_class(os)
    # print(os.m) #直接传入类会改变原创类的数据
    # try_class2(os.m)
    # print(os.m)
    # ktup = ('lin','')
    # if(ktup[0]=='lin'):
    #     print(ktup[0])
    # print(ktup[0])
    # dataSet = [[1]]
    # dataSet.append([4])
    # dataSet.append([7])
    # dataMat = mat(dataSet)
    dataSet2 = mat([[1,2,1],[4,5,6],[7,8,9]])
    dataSet2 = dataSet2[1,1]+1
    print(type(dataSet2))
    # print(dataSet2[[0,1],1]+1)
    # savetxt('try.txt',dataMat2)

    # dataMat2[[0,1],1]+=1    #返回的是一维数组
    # L = datamat3+1
    # print(dataMat2)
    # print(shape(dataMat))
    # dataMat.transpose()
    # print(mat(dataMat))
    # dataMul = multiply(dataMat,dataMat)
    # print(dataMul)
    # data2 = mat([[1,2],[3,4]])
    # data1 = mat([[5],[6]])
    # print(data2*data1)
    # print(multiply(data2,data1))
    # dataMat = mat([[1, 2], [3, 4], [5, 6]])
    # print(dataMat)

    # for data in dataSet_mat:
    #     print(data)
    # dataMat = loadDataSet('testSet.txt')
    # print('1')
    # print('2')
    # print('3')
    # print('4')
    # plotdata(dataSet_mat)
