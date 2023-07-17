"""
 作者 lgf
 日期 2023/5/15
"""
from keras.models import load_model  # TensorFlow is required for Keras to work
import numpy as np
import cv2
import tensorflow as tf


import requests
import os
import uuid
import pygame
import time
import pyttsx3


def tts_baidu(text,lang='zh',volume=0.5,speed=3):
    # 调用baiud网页截取MP3音频文件啊
    # 随机生成文件
    mp3_file = str(uuid.uuid4()) + '.mp3'
    flag = getaudio(text,mp3_file,speed=speed,type=lang)

    if flag:
        play_music(os.path.join("audios",mp3_file),volume=volume)
        # os.remove(os.path.join("audios",mp3_file))
    else:
        return False


def play_music(file,volume=0.5):
    pygame.mixer.init(frequency=44100)

    # 加载音乐
    pygame.mixer.music.load(file)
    pygame.mixer.music.set_volume(volume)
    pygame.mixer.music.play(start=0.0)

    # 播放时长，没有此设置，音乐不会播放，会一次性加载完
    while pygame.mixer.music.get_busy():
        pass

    pygame.mixer.quit()


def getaudio(content,saveto,speed=3,type='zh'):
    '''
    saveto:保存文件名称
    type:
    'en'英文,'zh'中文
    speed:语速
    '''
    url = 'https://fanyi.baidu.com/gettts'
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.72 Safari/537.36 Edg/89.0.774.45'
    }
    params = {
        'lan': type,  # 语言类型，zh中文，en英文
        'text': content,
        'spd': speed,  # 语速,经测试应填写1~6之内的数字
        'source': 'web',
    }
    if not os.path.exists('audios'):
        os.mkdir('audios')
    response = requests.get(url,params=params,headers=header)

    if response.status_code == 200 and response.content:  # 保存音频文件
        file_save = os.path.join("audios",saveto)

        with open(file_save,'wb') as f:
            f.write(response.content)
            return True
    else:

        raise Exception
        return False

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model(r'model\alphabet.h5', compile=False)

# Load the labels
class_names = open(r'model\alphabet.txt', "r",encoding='utf-8').readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)
set={''}

# 实时
# while True:
#     # Grab the webcamera's image.
#     ret, image = camera.read()
#     img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
#     # Make the image a numpy array and reshape it to the models input shape.
#     img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
#     # Normalize the image array
#     img = (img / 127.5) - 1
#     # Predicts the model
#     prediction = model.predict(img)
#     index = np.argmax(prediction)
#     class_name = class_names[index]
#     confidence_score = prediction[0][index]
#     # Print prediction and confidence score
#     print("Class:", class_name[2:])
#     set.add(class_name[2:])
#     cv2.putText(image, class_name[2:-1], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
#     cv2.imshow("Webcam Image", image)
#     # Listen to the keyboard for presses.
#     keyboard_input = cv2.waitKey(1)
#
#     # 27 is the ASCII for the esc key on your keyboard.
#     if keyboard_input == ord('q'):
#         break

# 文件夹
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
        set.add(class_name[2:])
        # 输出预测结果及置信度
        print("图片名：",filename)
        print("预测结果：",class_name[2:-1])
        print("置信度：",str(np.round(confidence_score * 100))[:-2],"%")

        # 将预测结果保存到 txt 文件中
        # with open(os.path.join('results',filename + '.txt'),'w') as f:
        #     f.write("预测结果：" + class_name[2:] + "\n")
        #     f.write("置信度：" + str(np.round(confidence_score * 100))[:-2] + "%\n")
txt=""
for x in set:
    x = x.replace('\n','').replace('\r','')
    txt+=x
print(txt)
pyttsx3.speak(txt)
camera.release()
cv2.destroyAllWindows()






