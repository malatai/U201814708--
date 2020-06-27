import operator
import datetime
import numpy as np
from numpy import *
from os import listdir
from skimage import io

print('程序处理的图片大小,建议不要超过200*200\n')
N = int(input('需要处理的图片的大小(100至200)，N='))
color = 100 / 255     


def KNN(test_data,train_data,train_label,k):
    dataSetSize = train_data.shape[0]
    all_distances = np.sqrt(np.sum(np.square(tile(test_data,(dataSetSize,1))-train_data),axis=1))
    sort_distance_index = all_distances.argsort()
    all_predictive_value = {}
    for i in range(k):
        predictive_value = train_label[sort_distance_index[i]]
        print('第',i+1,'次预测值',predictive_value)
        all_predictive_value[predictive_value] = all_predictive_value.get(predictive_value,0)+1
    sorted_class_count = sorted(all_predictive_value.items(), key = operator.itemgetter(1), reverse = True)
    return sorted_class_count[0][0]
def get_all_train_data():
    train_label = []
    train_file_list = listdir('trainlist')  
    m = len(train_file_list)             
    train_data = get_all_data(train_file_list,1)
    for i in range(m):
        file_name = train_file_list[i]      
        train_label.append(get_number_cut(file_name))  
    return train_label,train_data

def get_all_data(file_list,k):
    train_data = np.zeros([len(file_list), N**2])
    for i, item in enumerate(file_list):
        if k == 1:
            img = io.imread('./trainlist/'+ item, as_grey = True)
        else:
            img = io.imread('./testlist/' + item, as_grey = True)
        img[img>color] = 1
        img = get_cut_picture(img)
        img = get_stretch_picture(img).reshape(N**2)
        train_data[i, 0:N**2] = img
    return train_data

#切割图象
def get_cut_picture(img):
    size = []
    length = len(img)
    width = len(img[0,:])
    size.append(get_edge(img, length, 0, [-1, -1]))
    size.append(get_edge(img, width, 1, [-1, -1]))
    size = np.array(size).reshape(4)
    return img[size[0]:size[1]+1, size[2]:size[3]+1]

#获取切割边缘(高低左右的索引)
def get_edge(img, length, flag, size):
    for i in range(length):
        if flag == 0:
            line1 = img[img[i,:]<color]
            line2 = img[img[length-1-i,:]<color]
        else:
            line1 = img[img[:,i]<color]
            line2 = img[img[:,length-1-i]<color]
        if len(line1)>=1 and size[0]==-1:
            size[0] = i
        if len(line2)>=1 and size[1]==-1:
            size[1] = length-1-i
        if size[0]!=-1 and size[1]!=-1:
            break
    return size

#拉伸图像
def get_stretch_picture(img):
    newImg = np.ones(N**2).reshape(N, N)
    newImg1 = np.ones(N ** 2).reshape(N, N)
    step1 = len(img[0])/N
    step2 = len(img)/N
    for i in range(len(img)):
        for j in range(N):
            newImg[i, j] = img[i, int(np.floor(j*step1))]
    for i in range(N):
        for j in range(N):
            newImg1[j, i] = newImg[int(np.floor(j*step2)), i]
    return newImg1

def get_number_cut(file_name):
    fileStr = file_name.split('.')[0]           
    classNumStr = int(fileStr.split('_')[0])
    return classNumStr

#用字符矩阵打印图片
def get_show(test_data):
	for i in range(N**2):
		if(test_data[0,i] == 0):
			print ("1",end='')
		else:
			print ("0",end='')
		if (i+1)%N == 0 :
			print()

def main():
    Nearest_Neighbor_number = int(input('选取最邻近的K个值(建议小于7)，K='))
    train_label, train_data = get_all_train_data()

    test_file_list = listdir('testlist')
    file_name = test_file_list[0]
    test_index = get_number_cut(file_name)
    test_data = get_all_data(test_file_list,0)

    Result = KNN(test_data, train_data, train_label, Nearest_Neighbor_number)
    print ("最终预测值为:",Result,"    真实值:",test_index)

if __name__ == "__main__":
    main()