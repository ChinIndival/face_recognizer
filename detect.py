import cv2
import numpy as np
from PIL import Image
import pickle
import sqlite3
import random
from time import sleep
import os

# 顔検出器を初期化する
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

# 顔検出器を初期化する
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainner.yml')

def getProfile(id):
    conn=sqlite3.connect("FaceBaseNew.db")
    cursor=conn.execute("SELECT * FROM People WHERE ID="+str(id))
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

id=0
#文字スタイルを設定する
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0,255,0)
fontcolor1 = (0,0,255)

#写真から写真を取る
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
sleep(2)

print("写真を寄ってください")
while True:

    try:
        check, frame = webcam.read()
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(filename='avatar_images.jpg', img=frame)
            r = random.randint(1, 20000000)
            img_file = 'images' + str(r) + '.jpg'
            cv2.imwrite(filename='data/' + img_file, img=frame)
            webcam.release()
        
            img_ = cv2.imread('images.jpg', cv2.IMREAD_ANYCOLOR)
 
            cv2.destroyAllWindows()
            break

        elif key == ord('q'):
            webcam.release()
            cv2.destroyAllWindows()
            break

    except KeyboardInterrupt:
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break

image = cv2.imread('avatar_images.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_copy = np.copy(image)
 
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

faces = face_classifier.detectMultiScale(gray_image, 1.25, 6)

print('顔:', len(faces))

#保存した証明書の写真とUploadした写真を比較する
face_crop = []
for f in faces:
    x, y, w, h = [ v for v in f ]
    cv2.rectangle(image_copy, (x,y), (x+w, y+h), (255,0,0), 3)
    
    face_crop.append(gray_image[y:y+h, x:x+w])


for face in face_crop:
    cv2.imshow('face',face)
    cv2.waitKey(0)
    faces=faceDetect.detectMultiScale(face,1.3,5);

    for(x,y,w,h) in faces:
        id,dist=recognizer.predict(face)

        profile=None
        profile=getProfile(id)
        
        #情報を表示する
        if(profile!=None):
            print( "ID: " + str(profile[0]) + " | Name: " + str(profile[1]))
            f = open('text_detected'+str(profile[0])+'.txt', 'r')
            file_contents = f.read()
            print(file_contents)
            f.close()
            #認識できたとき、イメージにデータセットに追加する。
            r = random.randint(1, 20000000)
            cv2.imwrite("dataSet/User." + str(profile[0]) + '.' + str(r) + ".jpg", face)
            os.system('python3 TrainModel.py')
        else:
            print("No person")

    cv2.imshow('Face',face)
    # qを押すと終了
    if cv2.waitKey(1)==ord('q'):
        break;
