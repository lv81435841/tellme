"""
 作者 lgf
 日期 2023/3/26
"""

import os
from keras.models import load_model
import cv2
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model(r'C:\Users\小凡\Downloads\converted_keras\alphabet.h5',compile=False)
# Load the labels
class_names = open(r'C:\Users\小凡\Downloads\converted_keras\alphabet.txt',"r").readlines()

# 遍历指定文件夹下的所有图片文件
for filename in os.listdir('images'):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # 读取图片
        image = cv2.imread(os.path.join('images',filename))

        # 将图片缩放为指定尺寸
        image = cv2.resize(image,(224,224),interpolation=cv2.INTER_AREA)

        # 将图片转换为 numpy 数组并调整形状以适应模型的输入格式
        image = np.asarray(image,dtype=np.float32).reshape(1,224,224,3)

        # 对图片数组进行归一化处理
        image = (image / 127.5) - 1

        # 使用模型进行预测
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # 输出预测结果及置信度
        print("图片名：",filename)
        print("预测结果：",class_name[2:])
        print("置信度：",str(np.round(confidence_score * 100))[:-2],"%")

        # 将预测结果保存到 txt 文件中
        with open(os.path.join('results',filename + '.txt'),'w') as f:
            f.write("预测结果：" + class_name[2:] + "\n")
            f.write("置信度：" + str(np.round(confidence_score * 100))[:-2] + "%\n")