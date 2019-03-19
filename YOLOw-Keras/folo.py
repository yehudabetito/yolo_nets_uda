# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 12:34:58 2019

@author: Uda
"""
%reset -f
# folo 

# import the needed modules
import os
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
# The below provided fucntions will be used from yolo_utils.py
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes
# The below functions from the yad2k library will be used
from yad2k.models.keras_yolo import yolo_head, yolo_eval
# The below function from the yolo library will be used
from yolo_video import yolo_main

# i create automation for the yolo code that the maig goal is to take 
# as input k image and lists of k feature map 
# also from every image we want to take the boxes values of the segmaention
# and the class value of all the relevent object in the image

# some prep work
feature_map_folo = []; cordinate=[]; a_team = []
width_folo = []; height_folo = [];yolo_values = {} 
#now we want to call every relevent image
files =[];png_files = []
#append all the relvant image files in the folder 
#png_files.append(list(glob.iglob(r"C:\Users\Uda\YOLOw-Keras\images\*.png", recursive=True))) 
#files.append(list(glob.iglob(r"C:\Users\Uda\YOLOw-Keras\images\*.jpeg", recursive=True)))
files.append(list(glob.iglob(r"C:\Users\Uda\YOLOw-Keras\images\*.jpg", recursive=True)))


files[0][1]
a_team = [[] for iter in range(len(files))]
#yolo_values the first value is the feature map, the second on is the x1 y1 x2 y2
#the third one is the 
for i in range(len(files)): 
#for i in range(3): 

    yolo_values = yolo_main(chosen_image = os.path.basename(files[0][i]))
    feature_map_folo.append(yolo_values[0]) 
 #conditions that the box will make sense 
    cordinate.append(yolo_values[1])    
    cordinate[i][cordinate[i]<0] = 0
    cordinate[i][:,::2][cordinate[i][:,::2] > yolo_values[3][0]] = yolo_values[3][0]
    cordinate[i][:, 1::2][cordinate[i][:, 1::2] > yolo_values[3][1]] = yolo_values[3][1]
    
    
    
    if any(yolo_values[2]==59):
        j=np.where(yolo_values[2]==59)
        width_val,height_val = (cordinate[i][j,2:] - cordinate[i][j,0:2])  #calc the height and width  
        a_team[1].append( np.concatenate([cordinate[i][j,0:2] \
        ,[width_val,height_val]]))
    
    np.concatenate([cordinate[i][j,0:2] \,[width_val,height_val]])
    else:
        a_team[1].append(np.zeros(4))

b=np.array([2,3])
b=np.zeros((4100,24,1))
np.concatenate((a,b))
    np.concatenate((yolo_values[0], \
     np.concatenate([cordinate[i][j,0:2],[width_val,height_val]])))
yolo_values[0]


np.ravel(yolo_values[0]).shape

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
#    
# =============================================================================
# =============================================================================
    for j in range(len(cordinate[i])):# the number segmantion i yolo observed    
         width_val,height_val = (cordinate[i][j,2:] - cordinate[i][j,0:2])  #calc the height and width  
         a_team[1].append(np.concatenate([cordinate[i][j,0:2] \
         ,[width_val,height_val]]))
  
# =============================================================================
# convert png image to jpg image 
#so in the list files we neet to take only the png files and then convert tham to te
for i in range(len(png_files[0])):
    png_files[0][0]    
    im = Image.open(png_files[0][i])
    rgb_im = im.convert('RGB')
    rgb_im.save(os.path.splitext(os.path.basename(png_files[0][i]))[0]+'.jpg')
#



