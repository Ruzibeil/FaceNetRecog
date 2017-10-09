import tensorflow as tf
import numpy as np
import cv2

import os
from os.path import join as pjoin
import sys
import copy
import detect_face
import nn4 as network
import matplotlib.pyplot as plt


import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.externals import joblib

#%pylab inline

#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

#facenet embedding parameters

model_dir='./model_check_point/model.ckpt-500000'#"Directory containing the graph definition and checkpoint files.")
model_def= 'models.nn4'  # "Points to a module containing the definition of the inference graph.")
image_size=96 #"Image size (height, width) in pixels."
pool_type='MAX' #"The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn=False #"Enables Local Response Normalization after the first layers of the inception network."
seed=42,# "Random seed."
batch_size= None # "Number of images to process in a batch."

print('Creating networks and loading parameters')
gpu_memory_fraction=1.0
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')

#build facenet embedding model
print('build facenet embedding model')
tf.Graph().as_default()
sess = tf.Session()
images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 
                                                       image_size, 
                                                       image_size, 3), name='input')

phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')



embeddings = network.inference(images_placeholder, pool_type, 
                               use_lrn, 
                               1.0, 
                               phase_train=phase_train_placeholder)



ema = tf.train.ExponentialMovingAverage(1.0)
saver = tf.train.Saver(ema.variables_to_restore())

model_checkpoint_path='./model_check_point/model-20160506.ckpt-500000'

saver.restore(sess, model_checkpoint_path)
print('facenet embedding build over')

###### train_dir containing one subdirectory per image class 
#should like this:
#-->train_dir:
#     --->pic_me:
#            me1.jpg
#            me2.jpg
#            ...
#     --->pic_others:
#           other1.jpg
#            other2.jpg
#            ...
#data_dir='~/train_dir/'#your own train folder
data_dir='/home/dj/Downloads/real_time_face_recognition-master/train/'
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def read_img(person_dir,f):
    img=cv2.imread(pjoin(person_dir, f))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    if gray.ndim == 2:
        img = to_rgb(gray)
    return img

def load_data(data_dir):
    data = {}
    pics_ctr = 0
    for guy in os.listdir(data_dir):
        person_dir = pjoin(data_dir, guy)
        
        curr_pics = [read_img(person_dir, f) for f in os.listdir(person_dir)]
        
     
        
        data[guy] = curr_pics
        
    return data

data=load_data(data_dir)
keys=[]
for key in data.iterkeys():
    keys.append(key)
print keys
print keys[0]
print keys[1]
print keys[2]
print keys[3]
#print keys[4]


train_x=[]
train_y=[]

for x in data[keys[0]]:
    bounding_boxes, _ = detect_face.detect_face(x, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]#number of faces
  
    
    
    for face_position in bounding_boxes:
        face_position=face_position.astype(int)
        #print(face_position[0:4])
        cv2.rectangle(x, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
        crop=x[face_position[1]:face_position[3],
             face_position[0]:face_position[2],]
    
        crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )

        #print(crop.shape)
        
        crop_data=crop.reshape(-1,96,96,3)
        #print(crop_data.shape)
        
        emb_data = sess.run([embeddings], 
                            feed_dict={images_placeholder: np.array(crop_data), phase_train_placeholder: False })[0]
        
        
        train_x.append(emb_data)
        train_y.append(0)
print(len(train_x))

# for y in data[keys[1]]:
#     bounding_boxes, _ = detect_face.detect_face(y, minsize, pnet, rnet, onet, threshold, factor)
#     nrof_faces = bounding_boxes.shape[0]#number of faces
   
    
    
#     for face_position in bounding_boxes:
#         face_position=face_position.astype(int)
#         #print(face_position[0:4])
#         cv2.rectangle(y, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
#         crop=y[face_position[1]:face_position[3],
#              face_position[0]:face_position[2],]
    
#         crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )

#         crop_data=crop.reshape(-1,96,96,3)
#         #print(crop_data.shape)
#         emb_data = sess.run([embeddings], 
#                             feed_dict={images_placeholder: np.array(crop_data), phase_train_placeholder: False })[0]
        
        
#         train_x.append(emb_data)
#         train_y.append(1)
    
# print(len(train_x))

for z in data[keys[2]]:
    bounding_boxes, _ = detect_face.detect_face(z, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]#number of faces
   
    
    
    for face_position in bounding_boxes:
        face_position=face_position.astype(int)
        #print(face_position[0:4])
        cv2.rectangle(z, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
        crop=z[face_position[1]:face_position[3],
             face_position[0]:face_position[2],]
    
        crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )

        crop_data=crop.reshape(-1,96,96,3)
        #print(crop_data.shape)
        emb_data = sess.run([embeddings], 
                            feed_dict={images_placeholder: np.array(crop_data), phase_train_placeholder: False })[0]
        
        
        train_x.append(emb_data)
        train_y.append(2)
    
print(len(train_x))

for m in data[keys[3]]:
    bounding_boxes, _ = detect_face.detect_face(m, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]#number of faces
   
    
    
    for face_position in bounding_boxes:
        face_position=face_position.astype(int)
        #print(face_position[0:4])
        cv2.rectangle(m, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
        crop=m[face_position[1]:face_position[3],
             face_position[0]:face_position[2],]
    
        crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )

        crop_data=crop.reshape(-1,96,96,3)
        #print(crop_data.shape)
        emb_data = sess.run([embeddings], 
                            feed_dict={images_placeholder: np.array(crop_data), phase_train_placeholder: False })[0]
        
        
        train_x.append(emb_data)
        train_y.append(3)
    
print(len(train_x))




#train/test split
train_x=np.array(train_x)
train_x=train_x.reshape(-1,128)
train_y=np.array(train_y)
print(train_x.shape)
print(train_y.shape)


X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=.3, random_state=42)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

# KNN Classifier  
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)  
    return model  

classifiers = knn_classifier 

model = classifiers(X_train,y_train)  
predict = model.predict(X_test)  

accuracy = metrics.accuracy_score(y_test, predict)  
print ('accuracy: %.2f%%' % (100 * accuracy)  ) 
  
    
#save model
#joblib.dump(model, './model_check_point/knn_classifier.model')
joblib.dump(model, '/home/dj/Downloads/real_time_face_recognition-master/myknnmodel/knn_classifier5.model')
#model = joblib.load('_2017_1_24_knn.model')


model = joblib.load('/home/dj/Downloads/real_time_face_recognition-master/myknnmodel/knn_classifier5.model')
predict = model.predict(X_test) 
accuracy = metrics.accuracy_score(y_test, predict)  
print ('accuracy: %.2f%%' % (100 * accuracy)  )


