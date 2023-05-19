import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

fd=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
sd=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')
vid=cv2.VideoCapture(0)
notCaptured=True
seq=0
while notCaptured:
    flag,img=vid.read()
    if flag:
        # processing code
        #  img convert into gray image
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=fd.detectMultiScale(img_gray,1.1,5,minSize=(50,50))
        

        np.random.seed(50)
        colors=np.random.randint(0,255,(len(faces),3)).tolist()
                
        i=0
        for x,y,w,h in faces:
            face=img_gray[y:y+h,x:x+w].copy()
            smiles=sd.detectMultiScale(
                face,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50,50)
            )
            if len(smiles)==1:
                seq+=1
                print(seq)
                if seq ==10:
                   cv2.imwrite('myselfie.png',img)
                   notCaptured=False
                   break
            else:
                seq=0
            cv2.rectangle(
                img,pt1=(x,y),pt2=(x+w,y+h),color=colors[i],thickness=3
            )
            i+=1
        
       
        cv2.imshow('preview',img)
        key=cv2.waitKey(1)
        if key==ord('q'): ##for stop the camera ligth which is acquire by this program
           break
    else:
        print('No frames')
        break
    sleep(0.1)
vid.release()
cv2.destroyAllWindows()