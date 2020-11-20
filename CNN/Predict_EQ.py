#coding:utf-8

from keras.models import load_model
import matplotlib.image as processimage
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import sys


class Prediction(object):
    def __init__(self,ModelFile,PredictFile,EQType,Width=100,Height=100):
        self.modelfile = ModelFile
        self.predict_file = PredictFile
        self.Width = Width
        self.Height = Height
        self.EQType = EQType

    def Predict(self):
        #引入model
        model = load_model(self.modelfile)
        #处理照片格式和尺寸
        img_open = Image.open(self.predict_file)
        conv_RGB = img_open.convert('RGB')
        new_img = conv_RGB.resize((self.Width,self.Height),Image.BILINEAR)

        basedir = os.path.abspath(os.path.dirname(__file__))
        name = 'temp.jpeg'
        basepath = os.path.join(basedir,name)  #裁剪文件路径
        #print(basepath)
        orignpath = os.path.join(basedir,self.predict_file)  #原始路径
        #print(orignpath)

        new_img.save(basepath)#保存新的图片
        #print('Image Processed')
        #处理图片shape
        image = processimage.imread(basepath)
        image_to_array = np.array(image)/255.0#转成float
        image_to_array = image_to_array.reshape(-1,100,100,3)
        #print('Image reshaped')
        #预测图片
        prediction = model.predict(image_to_array)#prediction为[[    ]]
        #print(prediction)
        Final_prediction = [result.argmax() for result in prediction][0]
        #print(Final_prediction)

        result_list = []

        #延伸教程读取概率
        count = 0
        for i in prediction[0]:
            #print(i)
            percentage = '%.2f%%' % (i * 100)
            result_list.append(self.EQType[count]+percentage)
            print(self.EQType[count],'possibility:' ,percentage)
            count +=1

        #print(os.path.splitext(orignpath))
        result_str = '_'.join(result_list)      #预测结果
        (root,ext) = os.path.splitext(orignpath)  #('D:\\PycharmProjects\\EQ_train2\\b', '.jpeg')
        #os.path.join(root, ' '+result_str+ext)
        os.rename(orignpath, root+'_'+result_str+ext)   #将原始文件重命名
        #print(result_str)


    def ShowPredImg(self):
        image = processimage.imread(self.predict_file)
        plt.imshow(image)
        plt.show()


EQType = ['event', 'noise']
#实例化类
Pred = Prediction(PredictFile='b.jpeg',ModelFile='EQfinder.h5',Width=100,Height=100,EQType=EQType)

#Pred = Prediction(PredictFile='C:/Python/workspace/EQ_train/a.jpg',ModelFile='C:/Python/workspace/EQ_train/EQfinder.h5',Width=100,Height=100,EQType=EQType)
#Pred = Prediction(PredictFile=sys.argv[2],ModelFile=sys.argv[1],Width=100,Height=100,EQType=EQType)


#调用类
Pred.Predict()

