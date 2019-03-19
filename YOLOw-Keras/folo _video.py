# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 12:34:58 2019

@author: Uda
"""
# folo 

#Your statements here


#%reset -f
# import the needed modules
import os
import cv2
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
from PIL import Image
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import numpy as np
from PIL import ImageFilter,Image, ImageDraw, ImageFont, ImageEnhance, ImageChops
import glob
import timeit
# The below provided fucntions will be used from yolo_utils.py
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes

# The below functions from the yad2k library will be used
from yad2k.models.keras_yolo import yolo_head, yolo_eval
# The below function from the yolo library will be used
from yolo_video import yolo_main

start = timeit.default_timer()
  

# =============================================================================
# first lets adpet this code to work wiht videos as well
# =============================================================================

# i create automation for the yolo code that the maig goal is to take 
# as input k image and lists of k feature map 
# also from every image we want to take the boxes values of the segmaention
# and the class value of all the relevent object in the image
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.

      

# some prep work
cordinate=[]; width_folo = []; height_folo = [];yolo_values = []
a_team = [[] for iter in range(240)]
#we narrow down our classes to 20 from 80 so i created a dict that subset the 
#needed values
class_dict = {0:0,2:1,24:2,25:3,26:4,28:5,41:6,56:7,57:8,59:9\
      ,63:10,66:11,67:12,68:13,69:14,70:15,72:16,73:17,74:18,77:19}

one_movie = np.zeros((1,240,260))

#yolo_values the first value is the feature map, the second on is the x1 y1 x2 y2
#the third one is the 
#for i in range(len(files)): 

cap = cv2.VideoCapture(r'C:\Users\Uda\YOLOw-Keras\images\thats_the_one.avi')
if (cap.isOpened() == False):
   print("Unable to read video")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
count = 0
while (True) and count <25 :
   ret, frame = cap.read()   
   if ret == True: 
      cv2.imwrite("frame%d.jpg" % count, frame)  # save frame as JPEG file,somewhere
      
      #chosen_image = "frame%d.jpg" % count
      #frame_jpg = Image.open("frame%d.jpg" % count) # Opens the frame as JPEG file
      yolo_values = yolo_main(chosen_image = "frame%d.jpg" % count)
    #conditions that the box will make sense 
      cordinate.append(yolo_values[0])    
      cordinate[count][cordinate[count]<0] = 0
      cordinate[count][:,::2][cordinate[count][:,::2] > yolo_values[2][0]] = yolo_values[2][0]
      cordinate[count][:, 1::2][cordinate[count][:, 1::2] > yolo_values[2][1]] = yolo_values[2][1]
      OTT_vec = np.zeros((1,len(class_dict)))#the last 20 classes in the vector  represent the object we want to track
      if type(class_dict[77])==int:
            OTT_vec[:, class_dict[77]] = 1
      one_movie[0,count,239:-1] = OTT_vec
      num_det = np.array([])
      for j in range(len(cordinate[count])):
            width_val,height_val = (cordinate[count][j,2:] - cordinate[count][j,0:2])  #calc the height and width  
            a_team[j].append(list(map(int,(np.concatenate([cordinate[count][j,0:2] \
                         ,[width_val,height_val]])))))
            
            one_hot = np.zeros((1,len(class_dict)))           
            if type(class_dict[yolo_values[1][j]])==int:
                one_hot[:, class_dict[yolo_values[1][j]]] = 1
            one_hot.astype(int)
            num_det =  np.concatenate((num_det,np.concatenate((a_team[j][0],one_hot[0])))).astype(int)
      one_movie[0,count,0:24*len(cordinate[count])] = num_det
      count+=1
stop = timeit.default_timer()
print('Time: ', stop - start)


##now we want to call every relevent image
#files =[];png_files = []
##append all the relvant image files in the folder 
##png_files.append(list(glob.iglob(r"C:\Users\Uda\YOLOw-Keras\images\*.png", recursive=True))) 
##files.append(list(glob.iglob(r"C:\Users\Uda\YOLOw-Keras\images\*.jpeg", recursive=True)))
#files.append(list(glob.iglob(r"C:\Users\Uda\YOLOw-Keras\images\*.jpg", recursive=True)))


# i might not need that' we need to check it more deeply
# =============================================================================
#class_name = read_classes("model_data/coco_classes.txt")
#class_in_image=np.zeros((1,(len(class_name))))
#class_in_image[0,[yolo_values[2]]]=1
#         
#if (yolo_values[3][0]==1600 and yolo_values[3][1] == 900):
#        cordinate[i][:,::2][cordinate[i][:,::2] > 900] = 900  
#        cordinate[i][:, 1::2][cordinate[i][:, 1::2] > 1600] = 1600
#if yolo_values[3][1] == 1600 and yolo_values[3][0] == 900:
#        cordinate[i][:,::2][cordinate[i][:,::2]>1600] = 1600 
#        cordinate[i][:, 1::2][cordinate[i][:, 1::2]> 900]=900         
# we dont need this to lines either
#import ntpath
    
#ntpath.basename(os.path.basename(files[0][0]))
#for j in range(len(cordinate[i])):# the number segmantion i yolo observed
#        
#        width_folo.append((cordinate[i][0,2:] - cordinate[i][0,0:2])[0])  #calc the height and width  
#        height_folo.append((cordinate[i][0,2:] - cordinate[i][0,0:2])[1])
#        a_team[i].append(np.concatenate(( cordinate[i][0,0:2] ,width_folo[j] ,height_folo[j]),axis = None))

# convert png image to jpg image 
#so in the list files we neet to take only the png files and then convert tham to te
#for i in range(len(png_files[0])):
#    png_files[0][0]    
#    im = Image.open(png_files[0][i])
#    rgb_im = im.convert('RGB')
#    rgb_im.save(os.path.splitext(os.path.basename(png_files[0][i]))[0]+'.jpg')
#


