## REAL time FACE recognition

import cv2
import face_recognition as fr
import pandas as pd
import numpy as np
import webbrowser as wb
import os 

fd=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
vid=cv2.VideoCapture(0)
old_data=pd.read_csv('face_data.csv',index_col=0,sep='|')

while True:
    flag,img=vid.read()
    if flag:
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=fd.detectMultiScale(img_gray,scaleFactor=1.1,minNeighbors=5,minSize=(50,50))
        if len(faces)==1:
            x,y,w,h= faces[0]
            img_face= img[y:y+h,x:x+w,:].copy()
            img_face = cv2.resize(img_face,(400,400),cv2.INTER_CUBIC)
            face_encoding=fr.face_encodings(img_face)
            if len(face_encoding)==1:
                for ind,entries in old_data.iterrows():
                    matched=fr.compare_faces(
                        face_encoding,
                        np.array(eval(entries['encoding']))
                    )
                    confidence = 1-fr.face_distance(
                        face_encoding,
                        np.array(eval(entries['encoding']))
                    )
                    print(confidence)
                    if matched and confidence > 0.5:
                        print(entries['names'])
                        cv2.putText(
                            img,entries['names'],(30,30),
                            cv2.FONT_HERSHEY_COMPLEX, 1.5,(0,0,255),5
                        )
                        #  check this command pls ::::::
                        # if entries['names']== 'abhay':
                        #     wb.open_new( 'https://www.youtube.com/watch?v=PT2_F-1esPk')
                        break
                
            
        for x,y,w,h in faces:
            cv2.rectangle(img,pt1=(x,y),pt2=(x+w,y+h),color=(0,0,255),thickness=8)
        
        cv2.imshow('preview',img)
        key=cv2.waitKey(1)
        if key == ord('q'):
            break
    else:
        print("No frame")
        break

cv2.destroyAllWindows()
cv2.waitKey(1)
vid.release()