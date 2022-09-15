
import shutil
import os
import sys
from flask import Flask, render_template, request
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import PIL
from PIL import Image
import requests
from io import BytesIO
from PIL import ImageFilter
from PIL import ImageEnhance
from IPython.display import display
import time


# global b
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
   
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
       
        
        shutil.copy("D:\\bananaLeafidentification\\ref\\"+fileName, dst)
        image = cv2.imread("D:\\bananaLeafidentification\\ref\\"+fileName)
        # image=image.convert('RGB')
        img1 = Image.open("D:\\bananaLeafidentification\\ref\\"+fileName)
        img1=img1.convert('RGB')
        img1.save('D:\\bananaLeafidentification\\static\\images\\my.jpg')
        # exec(open("pixelvalueswork2.py").read())
        exec(open("pixelvalueswork3.py").read())



        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        image=np.zeros((300,300),dtype="uint8")
        retval2,threshold2 = cv2.threshold(gray_image,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.rectangle(threshold2,(10,10),(40,40),(0, 0, 255), 1)
        cv2.imwrite('static/threshold.jpg', threshold2)
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'healthyvsunhealthynew-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
                np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 4, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            # print('model loaded!')


        fig = plt.figure()
        diseasename=" "
        rem=" "
        rem1=" "
        str_label=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            
            if np.argmax(model_out) == 0:
                str_label = 'cordana'
            elif np.argmax(model_out) == 1:
                str_label = 'healthy'
            elif np.argmax(model_out) == 2:
                str_label = 'pestalotiopsis'
            elif np.argmax(model_out) == 3:
                str_label = 'sigatoka'
            
            
            
            if str_label == 'cordana':
                diseasename = "cordana "
                
            
                rem = "The remedies for cordana are:\n\n "
                rem1 = [" Discard or destroy any affected plants",  
                "Do not compost them.", 
                "Rotate your Banana plants yearly to prevent re-infection next year.", 
                "Use copper fungicites"]
            elif str_label == 'pestalotiopsis':
                diseasename = "pestalotiopsis"
                
                rem = "The remedies for pestalotiopsis are: "
                rem1 = [" Monitor the field, handpick diseased plants and bury them.",
                "Use sticky yellow plastic traps.", 
                "Spray insecticides such as organophosphates", 
                "carbametes during the seedliing stage.", "Use copper fungicites"]
            elif str_label == 'sigatoka':
                diseasename = "sigatoka"
                
                rem = "The remedies for sigatoka are: "
                rem1 = [" Monitor the field, handpick diseased plants and bury them.",
                "Use sticky yellow plastic traps.", 
                "Spray insecticides such as organophosphates",
                "carbametes during the seedliing stage.",
                "Use copper fungicites"]      
            elif str_label == 'Healthy':
                status= 'Healthy'

            
                
            return render_template('home.html', status=str_label, disease=diseasename, remedie=rem, remedie1=rem1, ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName, ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg", ImageDisplay2="http://127.0.0.1:5000/static/threshold.jpg",Segimage1="http://127.0.0.1:5000/static/croppedimage1.jpg",Segimage2="http://127.0.0.1:5000/static/croppedimage2.jpg",Segimage3="http://127.0.0.1:5000/static/croppedimage3.jpg",Dominantcolor="http://127.0.0.1:5000/static/Dominantcolor.jpg",Segimage4="http://127.0.0.1:5000/static/croppedimage4.jpg",Segimage5="http://127.0.0.1:5000/static/croppedimage5.jpg")

        return render_template('home.html')
if __name__ == '__main__':
    app.run(debug=True)
    
