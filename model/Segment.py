import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt
import sys
sys.path.append(r"E:\Work\pycharm_workplace\dcnn")
import utils.Getdata as Getdata
train_imgs,test_imgs,_=Getdata.load_voc("2012")
print(train_imgs)