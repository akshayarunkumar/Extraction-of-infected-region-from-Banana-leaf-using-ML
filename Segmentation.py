import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def getMaskedImage(name):
    
    im = cv2.imread('D:\\bananaLeafidentification\\ref\\robert.jpg')
    im=cv2.resize(im,(224,224))
  
    mask = cv2.imread('D:\\bananaLeafidentification\\ref\\robert.jpg')
    mask=cv2.resize(mask,(224,224))

    
    try:
        
        print(mask.shape)
        print(im.shape)
    except AttributeError:
        print('ff')
        
        #mask = cv2.imread('G:\\2022\\Projects\\Lung-Segmentation-in-TensorFlow-2.0-main\\2.png')
        
    
    masked = cv2.bitwise_and(im,im,mask=mask[:,:,0])

    
    masked = cv2.resize(masked,(512,512))
    
    label = int((name.split(".png")[0]).split("_")[-1])

    
    return(masked,label)
    
im, label = getMaskedImage("MCUCXR_0001_0")
cv2.imshow('Output',im)
cv2.waitKey(0)                            # Cleanup after any key is pressed
cv2.destroyAllWindows()
plt.imshow(im.astype(np.uint8))
print("Label: ",label)
