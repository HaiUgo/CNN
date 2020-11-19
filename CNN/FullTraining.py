#coding:utf-8
#coding: utf-8
#--**Created by Cao on 2019**--
#*****Main Trainning*****
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from PIL import Image
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dropout, MaxPooling2D,Dense,Activation
from keras.optimizers import Adam
from keras.utils import np_utils


#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        #self.val_loss = {'batch':[], 'epoch':[]}
        #self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        #self.val_loss['batch'].append(logs.get('val_loss'))
        #self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        #self.val_loss['epoch'].append(logs.get('val_loss'))
        #self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='Train ACC')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='Train Loss')
        # if loss_type == 'epoch':
        #     # val_acc
        #     plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
        #     # val_loss
        #     plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('ACC-Loss')
        plt.legend(loc="upper right")
        plt.savefig("train.png")
        plt.show()


#Pre process images
class PreFile(object):
    def __init__(self,FilePath,EQtype):
        self.FilePath = FilePath
        # Main dog folder is shared path can be submit to param of this class
        self.EQType = EQtype
        #the dogtype list is shared list between rename and resize fucntion

    def FileReName(self):
        count = 0
        for type in self.EQType: #对于波形类型，输出每个波形文件夹名称
            subfolder = os.listdir(self.FilePath+type)  # 列出文件夹中的所有文件
            for subclass in subfolder:  #output name of folder
                #print('count_classese:->>' , count)
                #print(subclass)
                #print(self.FilePath+type+'/'+subclass)
                os.rename(self.FilePath+type+'/'+subclass, self.FilePath+type+'/'+str(count)+'_'+subclass.split('.')[0]+".jpeg")
            count+=1

    def FileResize(self,Width,Height,Output_folder):
        for type in self.EQType:
            #print(type)
            files = os.listdir(self.FilePath+type)
            for i in files:
                img_open = Image.open(self.FilePath + type+'/' + i)
                conv_RGB = img_open.convert('RGB') #统一转换一下RGB格式 统一化
                new_img = conv_RGB.resize((Width,Height),Image.BILINEAR)
                new_img.save(os.path.join(Output_folder,os.path.basename(i)))

#main Training program
class Training(object):
    def __init__(self,batch_size,number_batch,categories,train_folder):
        self.batch_size = batch_size
        self.number_batch = number_batch
        self.categories = categories
        self.train_folder = train_folder

    #Read image and return Numpy array
    def read_train_images(self,filename):
        img = Image.open(self.train_folder+filename)
        return np.array(img)


    def train(self):
        train_img_list = []#图像列表
        train_label_list = []#对应标签列表
        for file in os.listdir(self.train_folder):
            files_img_in_array = self.read_train_images(filename=file)
            train_img_list.append(files_img_in_array) #Image list add up
            train_label_list.append(int(file.split('_')[0])) #lable list addup

        train_img_list = np.array(train_img_list)
        train_label_list = np.array(train_label_list)

        train_label_list = np_utils.to_categorical(train_label_list,self.categories) #format into binary [0,0,0,0,1,0,0]

        train_img_list = train_img_list.astype('float32')#转为浮点型
        train_img_list /= 255.0#进行归一化



        #-- setup Neural network CNN
        model = Sequential()
        #CNN Layer - 1
        model.add(Convolution2D(
            input_shape=(100, 100, 3),  # input shape ** channel last(TensorFlow)
            filters=32, #Output for next later layer output (100,100,32)
            kernel_size= (5,5) , #size of each filter in pixel
            padding= 'same', #边距处理方法 padding method
        ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(
            pool_size=(2,2), #Output for next layer (50,50,32)
            strides=(2,2),
            padding='same',#外边界处理
        ))

        #CNN Layer - 2
        model.add(Convolution2D(
            filters=64,  #Output for next layer (50,50,64)
            kernel_size=(3,3),
            padding='same',
        ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(  #Output for next layer (25,25,64)
            pool_size=(2,2),
            strides=(2,2),
            padding='same',
        ))

    #Fully connected Layer -1
        model.add(Flatten())#降维打击，降为一维，Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
        model.add(Dense(1024))
        model.add(Activation('relu'))
    # Fully connected Layer -2
        model.add(Dense(512))
        model.add(Activation('relu'))
    # Fully connected Layer -3
        model.add(Dense(256))
        model.add(Activation('relu'))
    # Fully connected Layer -4
        model.add(Dense(self.categories))
        model.add(Activation('softmax'))#分类
    # Define Optimizer
        # adam=Adam(lr=0.0005)#学习率
        sgd = keras.optimizers.SGD(lr = 0.001,decay=0.0005,momentum=0.9,nesterov=True)

    # Compile the model
        model.compile(optimizer=sgd,
                      loss="categorical_crossentropy",
                      metrics=['accuracy']
                      )

    # 创建一个实例
        history = LossHistory()

    # Fire up the network
        model.fit(
            x=train_img_list,
            y=train_label_list,
            epochs=self.number_batch,
            batch_size=self.batch_size,
            #validation_split=0.2,
            verbose=1,
            callbacks=[history],
        )
        #SAVE your work -model
        model.save('./EQfinder.h5')
        #model.save('EQfinder.h5')
		# 绘制acc-loss曲线
        history.loss_plot('epoch')

def MAIN():

    EQtype = ['event', 'noise']

    #****FILE Pre processing****
    #FILE = PreFile(FilePath='C:/Python/workspace/EQ_train/Raw_Img/',EQtype=EQtype)
    FILE = PreFile(FilePath='Raw_Img/',EQtype=EQtype)

    #****FILE Rename and Resize****
    FILE.FileReName()
    FILE.FileResize(Height=100,Width=100,Output_folder='train_img/')

    #Trainning Neural Network
    Train = Training(batch_size=32, number_batch=50, categories=2, train_folder='train_img/')
    Train.train()


if __name__ == "__main__":
    MAIN()






