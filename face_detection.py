import cv2
import numpy as np


cap=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
skip=0
i=0
face_data=[]
face_section=[]
dataset_path='/media/sanskar/Seagate Expansion Drive/Face Recognition/'
file_name=input('enter the name of person')
while True:
	ret,frame=cap.read()
	gray_frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
	
	if ret==False:
		continue
	faces=face_cascade.detectMultiScale(frame,1.3,5)
	faces=sorted(faces, key=lambda f:f[2]*f[3])

	for face in faces[-1:]:
		x,y,w,h=face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		offset=10
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		if face_section.size!=0:
			face_section = cv2.resize(face_section,(100,100))

		#skip+=1
		if face_section.size!=0:
			face_data.append(face_section)
			print(len(face_data))
			


	#face_section=np.array(face_section)
	#print(type(face_section))
	#print(face_section)
	face_section=np.array(face_section)
	cv2.imshow("video feed" , frame)
	if face_section.size!=0:
		cv2.imshow("face section" , face_section)
		
	key_pressed=cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break
	
	#i+=1
	#print('i is {}'.format(i))
face_data=np.array(face_data)
face_data=face_data.reshape(face_data.shape[0],-1)
print(face_data.shape)

np.save(dataset_path+file_name + '.npy',face_data)
print('file saved!')

cap.release()
cv2.destroyAllWindows()


