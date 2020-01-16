#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 04:27:01 2020

@author: pete
"""
import csv
import face_recognition
import os
from PIL import Image
import numpy as np
import sys
#import fastai
            
from scipy.cluster import hierarchy
import pandas as pd
import matplotlib.pyplot as plt

def face_distance1(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

#here you can experiment with diffrent measures
#    return np.linalg.norm(face_encodings - face_to_compare,1, axis=1)
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)
 
destdir = '/home/pete/face/cutsceleb'
#destdir = '/home/pete/facecomp1'
files = [ f for f in os.listdir(destdir) if os.path.isfile(os.path.join(destdir,f)) ] 

all_images=[]
predicted_classes=[]
k=0

#    for file in tests[:33]:
os.chdir(destdir)
k=0 
for file in files:
        k+=1
        print(k)
        known_obama_image = face_recognition.load_image_file(file)
        # Get the face encodings for the known images
        width=known_obama_image.shape[0]
        height=known_obama_image.shape[1]
        face_location=(0,width,height,0)
        face_locations=[face_location]
        obama_face_encoding = face_recognition.face_encodings(known_obama_image,face_locations)[0]
        if k==1:
            arrayf=obama_face_encoding           
        else:
            arrayf=np.vstack([ arrayf,obama_face_encoding])

df = pd.DataFrame(arrayf)
names=pd.DataFrame(files)
named=pd.concat([names, df],axis=1,ignore_index=True)
#named.to_csv('/home/pete/compare/dataframe4.csv', encoding='utf-8', index=False)
#df = pd.read_csv('embed1.csv', header=None)
#
#enter name of file with face
j=0
while True:
    try:
        # try code
        fname = input('Enter a filename with face: ')
        print(fname)
        kface=names.loc[names[0]==fname].index[0]
        zz=arrayf[kface]   
        break
    except:
        print("file does not exist, try another")
        j+=1
        if j>3 :    
            sys.exit()


test=face_distance1(arrayf, zz) #subtract from all row index of row with face
pdtest=pd.DataFrame(test)
distance=pd.concat([names, pdtest],axis=1,ignore_index=True)
sorted1=distance.sort_values(by=[1])

fig,ax = plt.subplots(1,8)
plt.figure(figsize=(999,999))
#plt.text(fontsize = 5)
#size = 200, 200
os.chdir(destdir)
for k in range(8):
    file=sorted1.iloc[k,0]
    dist1=sorted1.iloc[k,1]
    unknown_image = face_recognition.load_image_file(file)
    pil_image = Image.fromarray(unknown_image)
#    pil_image.thumbnail(size)
    ax[k].imshow(pil_image)
    ax[k].axis('off')
   
    img_name=file
    dist1="{0:.2f}".format(dist1)
    ax[k].set_title(img_name+dist1,fontsize = 8, rotation='vertical' )

#    ax[k].set_xlabel(dist1)

fig.show()
"""
complete2 = hierarchy.complete(df)
dn = hierarchy.dendrogram(complete2, labels=files)
#df = df.set_index(names)
#del df.index.name
fig = plt.figure()
#dn = hierarchy.dendrogram(complete,labels=names.names[0])
plt.show()
"""
