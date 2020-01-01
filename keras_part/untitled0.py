# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 15:14:01 2019

@author: Navein Kumar
"""

#import required library

from keras.models  import load_model

import numpy as np

import cv2



import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second


count=1
stabilize=0

prev=[]

occurence=[]
#load your model here MNIST-CNN.model

model = load_model("MNIST-CNN.model")


def CountFrequency(my_list): 
  
    # Creating an empty dictionary  
    freq = {} 
    for item in my_list: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
     

    global count
    if(count>9):
        MaxDictVal = max(freq, key=freq.get)
        del freq[MaxDictVal]
        
        MaxDictVal1 =max(freq, key=freq.get)
        
        if MaxDictVal1>MaxDictVal:
            MaxDictVal=MaxDictVal*10+MaxDictVal1
        
        else:
            MaxDictVal=MaxDictVal1*10+MaxDictVal
            
    else:
        MaxDictVal = max(freq, key=freq.get)
        
            

    return MaxDictVal      
  
    

def Sort(sub_li): 
    sub_li.sort(key = lambda x: x[0]) 
    return sub_li 

def check(no,c):
    global count
    
    if(no==count):
        count=count+1
        winsound.Beep(5000, duration)
    
    if(no!=count and no>count):
        winsound.Beep(frequency, duration)
        
    global stabilize
    
    stabilize=0
    
    global  occurence
    
    occurence=[]
        



#load  video

pred_no=[]

cap = cv2.VideoCapture(0)

while(True):

    #read frame frame from video

    ret, image = cap.read()
    
    img_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    
    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    
    # join my masks
    mask = mask0
    
    # set my output img to zero everywhere except my mask
    output_img = image.copy()
    output_img[np.where(mask==0)] = 0



    #perform basic operation to smooth image

    img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (5, 5), 0)



    #find threshold

    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)



    #find contours and draw contours

    _,ctrs,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    
    
    fin_ctrs=[]
    
    for ctr in ctrs:
        area = cv2.contourArea(ctr)
        
        if(area>1000):
            fin_ctrs.append(ctr)
            
    cv2.drawContours(image,fin_ctrs,-1,(255,255,0),2)        
        

    rects = [cv2.boundingRect(ctr) for ctr in fin_ctrs]
    
    no_detected=""
    
    
    
    rectss=Sort(rects)
    
    
  
 
    
    
    prev=pred_no
    pred_no=[]
    for rect in rectss:
        
       
        
        
        
        

        x,y,w,h = rect

        if  h > 50 and h < 300  or w > 10 :

            #draw rectangel on image

            cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

            leng = int(rect[3] * 1.6)

            pt1 = abs(int(rect[1] + rect[3] // 2 - leng // 2))

            pt2 = abs(int(rect[0] + rect[2] // 2 - leng // 2))

            roi = img[pt1:pt1+leng, pt2:pt2+leng]

            roi = cv2.resize(roi,(28, 28), interpolation=cv2.INTER_AREA)

            #resize image

            roi = roi.reshape(-1,28, 28, 1)

            roi = np.array(roi, dtype='float32')

            roi /= 255

            pred_array = model.predict(roi)
            
            index=np.argmax(pred_array)
            
            
                

            pred_array = np.argmax(pred_array)
            

            #print('Result: {0}'.format(pred_array))

            cv2.putText(image, str(pred_array), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
            
            pred_no.append(pred_array)
            
            #no_detected.join(pred_array)

    #show frame
    
    
    
    if(pred_no==prev and pred_no!=[]):
        stabilize=stabilize+1
        occurence.extend(pred_no)
    
    

    if((pred_no!=[])):
       
       s = [str(i) for i in pred_no] 
       res = int("".join(s))
       print(res)
       if(stabilize>200):
           
           if(res==10 or res==21 or res==20):
               print("The predictted value is ",res,"The expected value is ",count)
               check(res,count)
               
           else:
               
               confident=CountFrequency(occurence)
               print("The predictted value is ",confident,"The expected value is ",count)
               check(confident,count)
    
    cv2.imshow("Result",image)

   

    k = cv2.waitKey(27)

    if k==27:

        break

cv2.destroyAllWindows()

cap.release()

