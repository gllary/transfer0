# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 19:37:38 2018

@author:GYX
"""

import glob
import os.path
import random
import numpy as np
import tensorflow as tf 
from tensorflow.python.platform import gfile

#Inception -v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

#Inception -v3模型中代表瓶颈层结果的张量的名称。在谷歌提供的Inception-v3模型中能够，这个张量的名字
#就是‘pool_3/_reshape:0'.在训练的模型时，可以通过tensor.name获取张量的名字
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

#图像输入张量所对应的名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

#下载的谷歌训练好的Inception-v3模型文件目录
MODEL_DIR = 'D:/transfer'

#下载好的Inception-v3模型文件名
MODEL_FILE = 'classify_image_graph_def.pb'

#因为一个训练数据可以被使用很多次，所以可以将原始图像通过Inception-v3模型计算得到
#的特征向量保存在文件中，免去重复计算。下面的变量定义了这些文件的存放地址
CACHE_DIR = '/tmp/bottleneck'

#图片数据文件夹。在这个文件夹中的每个子文件夹代表一个需要区分的类别，每个子文件夹中
#存放了相应类别的图片
INPUT_DATA = 'D:/transfer/flower_data'

#验证的数据百分比
VALIDATION_PERCENTAGE = 10
#测试的数据百分比
TEST_PERCENTAGE = 10

#定义神经网络的设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

#这个函数从数据文件夹中读取所有图片列表并按训练、验证、测试数据分开
#testing_percentage 和validation_percentage参数制定了测试数据集和验证数据集的大小
def creat_image_lists(testing_percentage, validation_percentage):
    #得到的所有图片都存在result这个字典中。这个字典的key为类别名
    #value也是一个字典，字典里面储存了所有图片名称
    result = {}
    #获取当前目录下所有子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    #得到第一个目录是当前目录，不需要考虑
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        #获取当前目录下所有的有效图片
        extensions = ['jpg','jpeg','JPG','JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:continue
        #通过目录名获取类别名称
        label_name = dir_name.lower()
        #初始化当前类别的训练数据集，测试数据集和验证数据集
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            #随机将数据分到训练数据集、测试数据集和验证数据集
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir':dir_name,
            'training':training_images,
            'testing':testing_images,
            'validation':validation_images,
        }
    #返回整理好的所有数据
    return result

#这个函数通过类别名称、所属数据集和图片编号获取一张图片的地址
#image_lists 参数给出了所有图片信息
#image_dir 参数给出了根目录。存放图片数据和存放图片特征向量的根目录地址不同
#label_name参数给定了类别的名称
#index参数给定了需要获取的图片的编号
#category参数制定了需要获取的图片是在训练数据集、测试数据集还是验证数据集上
def get_image_path(image_lists, image_dir,label_name,index,category):
    #获取给定类别中所有图片的信息
    label_lists = image_lists[label_name]
    #根据所属的数据集名称获取集合中的全部图片信息
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    #获取图片的文件名
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    #最终的地址为数据根目录的地址加上类别文件夹加上图片名称
    full_path = os.path.join(image_dir,sub_dir,base_name)
    return full_path

#此函数通过类别名称、所属数据集和图片编号获取经过Inception-v3模型处理之后的特征向量
#文件地址
def get_bottleneck_path(image_lists,label_name, index,category):
    return get_image_path(image_lists,CACHE_DIR,label_name,index,category)+'.txt'

#此函数使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量
def run_bottleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
    #这个过程实际上就是将当前图片作为输入计算瓶颈张量的值。这个瓶颈张量的值窥视这张图片
    #新的特征向量
    bottleneck_values = sess.run(bottleneck_tensor,{image_data_tensor:image_data})
    #经过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个特征向量（一维ie）
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

#此函数获取一张图片进过Inception-v3处理之后的特征向量。这个函数会先试图寻找已经计算且保存
#下来的特征像力量，如果找不到则先计算这个特征向量，然后保存到文件
def get_or_create_bottleneck(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor):
    #获取一张图片对应的特征向量文件路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR,sub_dir)
    if not os.path.exists(sub_dir_path):os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists,label_name,index,category)
    #如果这个特征向量文件不存在，则通过Inception-v3模型来计算特征向量，宁江计算结果寻如文件
    if not os.path.exists(bottleneck_path):
        #获取原始的图片路径
        image_path = get_image_path(image_lists,INPUT_DATA,label_name,index,category)
        #获取图片内容
        image_data = gfile.FastGFile(image_path,'rb').read()
        #通过Inception-v3模型计算特征向量
        bottleneck_values = run_bottleneck_on_image(sess,image_data,jpeg_data_tensor,bottleneck_tensor)
        #计算得到的特征向量存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path,'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        #直接从文件中获取图片相应的特征向量
        with open(bottleneck_path,'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    #返回得到的特征向量
    return bottleneck_values

#此函数随机获取一个batch的图片作为训练数据
def get_random_cached_bottleneck(sess,n_classes,image_lists,how_many,category,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        #随机一个类别和图片的编号加入当前的训练数据
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(sess,image_lists,label_name,image_index,category,jpeg_data_tensor,bottleneck_tensof)
        ground_truth = np.zeros(n_classes,dtype=np.float32)
        ground_truth[label_index]=1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks,ground_truths

#此函数获取全部的测试数据。在最终测试的时候需要在所有的测试数据上计算正确率
def get_test_bottlenecks(sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    #枚举所有的类别和每个类别中的测试图片
    for label_index,label_name in enumerate(label_name_list):
        category = 'testing'
        for index,unused_base_name in enumerate(image_lists[label_name][category]):
            #通过Inception-v3模型计算图片对应的特征向量，并将其加入最终数据的列表
            bottleneck = get_or_create_bottleneck(sess,images_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor)
            ground_truth = np.zeros(n_classes,dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks,ground_truths
    
def main():
    #读取所有图片
    image_lists = create_image_lists(TEST_PERCENTAGE,VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    #读取已经训练的Inception-v3模型。谷歌训练好的模型保存在了GraphDefProtocol
    #Buffer中，里面保存了每一个节点取值计算方法以及变脸的取值。TensorFlow模型持久
    #的问题在第五章中有详细发介绍
    with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    #加载读取的Inception-v3模型，并返回数据输入所对应的张量以及计算瓶颈结果所对应的张量
    bottleneck_tensor,jpeg_data_tensor = tf.improt_graph_def(graph_def,return_elements=[BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME])
    #定义新的标准答案输入
    ground_truth_input=tf.placeholder(tf.float32,[None,n_classes],name='GroundTruthInput')
    #定义一层全连接层来解决新的图片分类问题，因为训练好的Inception-v3模型已经将原始
    #的图片抽象为了更佳容易分类的特征向量了，所以不需要在训练那么复杂的神经网络来实现
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE,n_classes],stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input,weights)+biases
        final_tensor = tf.nn.softmax(logits)
    #定义交叉熵损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,ground_truth_input)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    
    #计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor,1),tf.argmax(ground_truth_input,1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        
    #训练过程
    for i in range(STEPS):
        #每次获取一个bacth的训练数据
        train_bottlenecks,train_ground_truth=get_random_cached_bottlenecks(sess,n_classes,image_lists,BATCH,'training',jpeg_data_tensor,bottleneck_tensor)
        sess.run(train_step,feed_dict={bottleneck_input:train_bottlenecks,ground_truth_input:train_ground_truth})
        #在验证数据上测试正确率
        if i%100 ==0 or i+1 == STEPS:
            validation_bottlenecks,validation_ground_truth = get_random_cached_bottlenecks(sess,n_classes,image_lists,BATCH,'validation',jpeg_data_tensor,bottleneck_tensor)
            validation_accuracy = sess.run(evaluation_step,feed_dict={bottleneck_input:validation_bottlenecks,ground_truth_input:validation_ground_truth})
            print('Step %d: Validation accuracy on random sampled %d examples= %.lf%%' % (i,BATCH,validation_accuracy*100))
        #在最后的测试数据上测试正确率
        test_bottlenecks,test_ground_truth = get_test_bottlenecks(sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step,feed_dict={bottleneck_input:test_bottlenecks,ground_truth_input:test_ground_truth})
        print('Final test accuracy = %.lf%%' % (test_accuracy *100))
if __name__ == '__main__':
    tf.app.run()