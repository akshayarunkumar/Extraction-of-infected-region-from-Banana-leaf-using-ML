import PIL
from PIL import Image
import requests
from io import BytesIO
from PIL import ImageFilter
from PIL import ImageEnhance
from IPython.display import display
import numpy as np
from tqdm import tqdm, trange
import time

st = time.time()
import cv2


pbar = tqdm(total=100)
IMG_SIZE=255
def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)

img = cv2.imread('D:\\bananaLeafidentification\\static\\images\\my.jpg')
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
height, width, _ = np.shape(img)
# print(height, width)

data = np.reshape(img, (height * width, 3))
data = np.float32(data)

number_clusters = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(data, number_clusters, None, criteria, 10, flags)
# print(centers)

font = cv2.FONT_HERSHEY_SIMPLEX
bars = []
rgb_values = []

for index, row in enumerate(centers):
    bar, rgb = create_bar(200, 200, row)
    bars.append(bar)
    rgb_values.append(rgb)


img_bar = np.hstack(bars)
lis=[]
for index, row in enumerate(rgb_values):
    image = cv2.putText(img_bar, f'{index + 1}. RGB: {row}', (5 + 200 * index, 200 - 10),
                        font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    # print(f'{index + 1}. RGB{row}')
    # print('test')
    lis.append(row)
print(lis[0])
txt=lis[0]
aa=list(txt)
rtresh=aa[0]
gtresh=aa[1]
btresh=aa[2]
print(rtresh)
print(gtresh)
print(btresh)


txt1=lis[1]
aa1=list(txt1)
rtresh1=aa1[0]
gtresh1=aa1[1]
btresh1=aa1[2]
print(rtresh1)
print(gtresh1)
print(btresh1)


txt2=lis[2]
aa2=list(txt2)
rtresh2=aa2[0]
gtresh2=aa2[1]
btresh2=aa2[2]
print(rtresh2)
print(gtresh2)
print(btresh2)

txt4=lis[3]
aa4=list(txt4)
rtresh4=aa4[0]
gtresh4=aa4[1]
btresh4=aa4[2]
print(rtresh4)
print(gtresh4)
print(btresh4)


txt5=lis[4]
aa5=list(txt5)
rtresh5=aa5[0]
gtresh5=aa5[1]
btresh5=aa5[2]
print(rtresh5)
print(gtresh5)
print(btresh5)
# x = txt.split(",")
# print(x)
# cv2.imshow('Image', img)
# cv2.imshow('Dominant colors', img_bar)
filename = 'Dominantcolor.jpg'
  
# Using cv2.imwrite() method
# Saving the image

cv2.imwrite('static/Dominantcolor.jpg', img_bar)
# cv2.imwrite('output/bar.jpg', img_bar)

cv2.waitKey(0)

# rtresh=92
# gtresh=59
# btresh=39
img = Image.open(r"D:\\bananaLeafidentification\\static\\images\\my.jpg")
img = img.resize((255, 255)) 
print(img.mode)
width, height = img.size

print(height)
print(width)
print(img.getpixel((228,2)))
x,y,z=img.getpixel((228,2))
# print(x)
# print(y)
# print(z)
lisx=[]
lisy=[]
for xi in range(width-1):
    for yj in range(height-1):
        x,y,z=img.getpixel((xi,yj))
        if(x>rtresh-20 and x<rtresh+20 and y>gtresh-20 and y<gtresh+20 and z>btresh-20 and z<btresh+20):
            # print(xi,yj)
            # print("\n")
            lisx.append(xi)
            lisy.append(yj)
print(min(lisx))
print(min(lisy))
print(max(lisx))
print(max(lisy))
print("20% Done")
pbar.update(20)
# print(lisx)
from PIL import Image
img1 = Image.open('D:\\bananaLeafidentification\\static\\images\\my.jpg')
img1 = img1.resize((255, 255)) 
img1.save("originalimage.jpg")
# img1.show()
box = (min(lisx), min(lisy), max(lisx), max(lisy))
img2 = img1.crop(box)
img2.save("D:\\bananaLeafidentification\\static\\croppedimage1.jpg")
# img2.show()


# for second image
lisx1=[]
lisy1=[]
for xi1 in range(width-1):
    for yj1 in range(height-1):
        x1,y1,z1=img.getpixel((xi1,yj1))
        if(x1>rtresh1-20 and x1<rtresh1+20 and y1>gtresh1-20 and y1<gtresh1+20 and z1>btresh1-20 and z1<btresh1+20):
            # print(xi1,yj1)
            # print("\n")
            lisx1.append(xi1)
            lisy1.append(yj1)
print(min(lisx1))
print(min(lisy1))
print(max(lisx1))
print(max(lisy1))
# print(lisx)
from PIL import Image
img1 = Image.open('D:\\bananaLeafidentification\\static\\images\\my.jpg')
img1 = img1.resize((255, 255)) 
# img1.show()
box = (min(lisx1), min(lisy1), max(lisx1), max(lisy1))
img2 = img1.crop(box)
img2.save("D:\\bananaLeafidentification\\static\\croppedimage2.jpg")
# img2.show()
print("40% Done")
pbar.update(20)

# for third image
lisx2=[]
lisy2=[]
for xi2 in range(width-1):
    for yj2 in range(height-1):
        x2,y2,z2=img.getpixel((xi2,yj2))
        if(x2>rtresh2-20 and x2<rtresh2+20 and y2>gtresh2-20 and y2<gtresh2+20 and z2>btresh2-20 and z2<btresh2+20):
            # print(xi1,yj1)
            # print("\n")
            lisx2.append(xi2)
            lisy2.append(yj2)
print(min(lisx2))
print(min(lisy2))
print(max(lisx2))
print(max(lisy2))
# print(lisx)
from PIL import Image
img1 = Image.open('D:\\bananaLeafidentification\\static\\images\\my.jpg')
img1 = img1.resize((255, 255)) 
# img1.show()
box = (min(lisx2), min(lisy2), max(lisx2), max(lisy2))
img2 = img1.crop(box)
img2.save("D:\\bananaLeafidentification\\static\\croppedimage3.jpg")
# img2.show()
print("60% Done")
pbar.update(20)

# for fourth image
lisx4=[]
lisy4=[]
for xi4 in range(width-1):
    for yj4 in range(height-1):
        x4,y4,z4=img.getpixel((xi4,yj4))
        if(x4>rtresh4-20 and x4<rtresh4+20 and y4>gtresh4-20 and y4<gtresh4+20 and z4>btresh4-20 and z4<btresh4+20):
            # print(xi1,yj1)
            # print("\n")
            lisx4.append(xi4)
            lisy4.append(yj4)
print(min(lisx4))
print(min(lisy4))
print(max(lisx4))
print(max(lisy4))
# print(lisx)
from PIL import Image
img1 = Image.open('D:\\bananaLeafidentification\\static\\images\\my.jpg')
img1 = img1.resize((255, 255)) 
# img1.show()
box = (min(lisx4), min(lisy4), max(lisx4), max(lisy4))
img2 = img1.crop(box)
img2.save("D:\\bananaLeafidentification\\static\\croppedimage4.jpg")
# img2.show()
print("80% Done")
pbar.update(20)

# for fifth image
lisx5=[]
lisy5=[]
for xi5 in range(width-1):
    for yj5 in range(height-1):
        x5,y5,z5=img.getpixel((xi5,yj5))
        if(x5>rtresh5-20 and x5<rtresh5+20 and y5>gtresh5-20 and y5<gtresh5+20 and z5>btresh5-20 and z5<btresh5+20):
            # print(xi1,yj1)
            # print("\n")
            lisx5.append(xi5)
            lisy5.append(yj5)
print(min(lisx5))
print(min(lisy5))
print(max(lisx5))
print(max(lisy5))
# print(lisx)
from PIL import Image
img1 = Image.open('D:\\bananaLeafidentification\\static\\images\\my.jpg')
img1 = img1.resize((255, 255)) 
# img1.show()
box = (min(lisx5), min(lisy5), max(lisx5), max(lisy5))
img2 = img1.crop(box)
img2.save("D:\\bananaLeafidentification\\static\\croppedimage5.jpg")
# img2.show()
print("100% Done")
pbar.update(20)
pbar.close()
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
elapsed_time=0
et=0
st=0
# cv2.waitKey(0)