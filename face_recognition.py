import numpy as np
import pandas as pd 
import os
import sys
import cv2
import csv
from datetime  import datetime


def distance (v1,v2):
	return np.sqrt(((v1-v2)**2)).sum()

def knn(train,test,k=5):
	dist=[]

	for i in range (train.shape[0]):
		ix=train[i,:-1]
		iy=train[i,-1]

		d=distance(test,ix)
		dist.append([d,iy])
	d_k=sorted(dist, key=lambda x: x[0])[:k]
	labels=np.array(d_k)[:,-1]

	output=np.unique(labels, return_counts=True)

	index=np.argmax(output[1])
	return output[0][index]
###############################################################################################
cap=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
skip=0
i=0
face_data = []
face_section = []
labels = []
dataset_path='/media/sanskar/Seagate Expansion Drive/Face Recognition/'
class_id=0
names={}
csv_name=[]
csv_time=[]
#data preperation 

for f in os.listdir(dataset_path):
	#print(f[-1:-4])
	if f.endswith('.npy'):
		names[class_id]=f[:-4]
		data_item=np.load(dataset_path + f)
		print("shape is" , data_item.shape)
		face_data.append(data_item)
		print('loaded ' + f)
		target = class_id*np.ones((data_item.shape[0],))
		class_id+=1
		labels.append(target)
		print(labels)
		
x=np.array(face_data)
print(x.shape)
face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels, axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)
exit(0);

#Testing
while True:
	ret,frame=cap.read()

	if ret==False:
		continue

	faces=face_cascade.detectMultiScale(frame,1.3,5)

	for face in faces[-1:]:
		x,y,w,h=face

		offset=10
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]

		if face_section.size!=0:
			face_section = cv2.resize(face_section,(100,100))

		out = knn(trainset,face_section.flatten())

		pred_name=names[int(out)]
		cv2.putText(frame , pred_name, (x,y-10) , cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,255) , 2)
		if pred_name not in csv_name:
			csv_name.append(pred_name)
			csv_time.append(datetime.now())

	cv2.imshow('Faces' , frame)

	key=(cv2.waitKey(1) & 0xFF)
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

print(csv_name , csv_time)

data={
	'Name':csv_name,
	'Time' : csv_time 
}

df=pd.DataFrame(data)
x=datetime.now()
df.to_csv('Attendance{}'.format(x.strftime('%x')))