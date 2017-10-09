import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
import sys
import copy
import detect_face
import nn4 as network
import random


import sklearn

from sklearn.externals import joblib
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




frame_interval=3 # frame intervals
def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret
print('Creating networks and loading parameters')
gpu_memory_fraction=1.0
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')
#restore facenet model
print('facenet embedding model')
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
#ckpt = tf.train.get_checkpoint_state(os.path.expanduser(model_dir))
#saver.restore(sess, ckpt.model_checkpoint_path)

model_checkpoint_path='./model_check_point/model-20160506.ckpt-500000'
#ckpt = tf.train.get_checkpoint_state(os.path.expanduser(model_dir))
#model_checkpoint_path='model-20160506.ckpt-500000'


#saver.restore(sess, ckpt.model_checkpoint_path)
saver.restore(sess, model_checkpoint_path)
print('facenet embedding model done')
#restore pre-trained knn classifier
model = joblib.load('./model_check_point/knn_classifier5.model')
#obtaining frames from camera--->converting to gray--->converting to rgb
#--->detecting faces---->croping faces--->embedding--->classifying--->print


video_capture = cv2.VideoCapture(0)
# size = (int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
#         int(video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
# out = cv2.VideoWriter('/home/dj/Downloads/output.avi',cv2.cv.CV_FOURCC(*'DIVX'), 10.0, size,1)
c=0
 
while True:
    # Capture frame-by-frame

    ret, frame = video_capture.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(frame.shape)
    
    timeF = frame_interval
    
    
    if(c%timeF == 0): #frame_interval==3, face detection every 3 frames
        
        find_results=[]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        if gray.ndim == 2:
            img = to_rgb(gray)
        
            

        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        
        
        nrof_faces = bounding_boxes.shape[0]#number of faces
        
        

        for face_position in bounding_boxes:
            
            face_position=face_position.astype(int)
            
            #print((int(face_position[0]), int( face_position[1])))
            #word_position.append((int(face_position[0]), int( face_position[1])))
           
            cv2.rectangle(frame, (face_position[0], 
                            face_position[1]), 
                      (face_position[2], face_position[3]), 
                      (0, 255, 0), 2)
            
            crop=img[face_position[1]:face_position[3],face_position[0]:face_position[2],]
    
            crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )
        
            data=crop.reshape(-1,96,96,3)
        
            emb_data = sess.run([embeddings], 
                                feed_dict={images_placeholder: np.array(data), 
                                           phase_train_placeholder: False })[0]
      #      print emb_data
            predict = model.predict(emb_data)
            print predict 
         
       
            if predict==0:
                find_results.append('zhangjiao')
               
            elif predict==1:
                find_results.append('others')
                 
            elif predict==2:
                find_results.append('dingjie')
            elif predict==3:
                find_results.append('buwei')       
            # else:
            #     find_results.append('others')

            print find_results
    
 
        cv2.putText(frame,'detected:{}'.format(find_results), (50,100), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0 ,0), 
                thickness = 2, lineType = 2)
  
            
    #print(faces)
    c+=1
    # Draw a rectangle around the faces
    


    # Display the resulting frame
 #   out.write(frame)
    cv2.imshow('Video', frame)
 #   formated_str="%d.png"%(c)
 #   cv2.imwrite(formated_str,frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


# When everything is done, release the capture

video_capture.release()
out.release()
cv2.destroyAllWindows()
