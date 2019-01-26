#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kevin Di'

import os
import random 
 
# VOC like data_set file path.

xml_file = r'path to your VOC like data_set: /Annotations'
img_file = r'path to your VOC like data_set:/JPEGImages'
save_path = r'path to your VOC like data_set: /ImageSets/Main'


# Determine the train, val, test split ratio.
# The frist step is split the train_val and test, and then split the train and val from the train_val.

train_val_percent = 0.8
train_percent = 0.8
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
