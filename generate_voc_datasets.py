#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kevin Di'

import os
import random 
 
# xml文件及Main文件夹位置。

xml_file = r'/Users/kevin/Documents/Tensorflow/container_detection/data_set/cont_voc/Annotations'
img_file = r'/Users/kevin/Documents/Tensorflow/container_detection/data_set/cont_voc/JPEGImages'
save_path = r'/Users/kevin/Documents/Tensorflow/container_detection/data_set/cont_voc/ImageSets/Main'

# 确定train、val、test拆分比例，先分拆出来train_val与test，再从train_val中分拆出train与val。

train_val_percent = 1
train_percent = 0.9
total_dataset_num = os.listdir(xml_file)
total_img_num = os.listdir(img_file)
num = len(total_dataset_num)
img = len(total_img_num)
list = range(num)  
t_v = int(num * train_val_percent)  
t = int(t_v * train_percent)  
train_val= random.sample(list,t_v)  
train = random.sample(train_val,t)  
 
print('Total number of  xml files is:', num)
print('Total number of images is:', img)
print('training set size:', t)
print('validation set size:', t_v - t)
print('test set size:', num - t_v)

file_train = open(os.path.join(save_path,'train.txt'), 'w') 
file_val = open(os.path.join(save_path,'val.txt'), 'w')  
file_test = open(os.path.join(save_path,'test.txt'), 'w')  
 
 
for i in list:  
    xml_name = total_dataset_num[i][:5]+'\n'

    if i in train_val:  
        if i in train:  
            file_train.write(xml_name)  
        else:  
            file_val.write(xml_name)  
    else:  
        file_test.write(xml_name)  
  
file_train.close()  
file_val.close()  
file_test.close()
