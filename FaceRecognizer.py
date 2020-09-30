import cv2
import numpy as np
from PIL import Image
import pickle
import sqlite3

# 顔検出器を初期化する
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

# 顔検出器を初期化する
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainner.yml')



id=0
#文字スタイルを設定する
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0,255,0)
fontcolor1 = (0,0,255)

# IDでユーザー情報を取得する
def getProfile(id):
    conn=sqlite3.connect("FaceBaseNew.db")
    cursor=conn.execute("SELECT * FROM People WHERE ID="+str(id))
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

# カメラを初期化する
cam=cv2.VideoCapture(0);

while(True):

    # カメラから写真を読む
    ret,img=cam.read();

    # 画像を上下逆にします
    img = cv2.flip(img, 1)

    # 長方形のフレームを描いて顔を配置します
    centerH = img.shape[0] // 2;
    centerW = img.shape[1] // 2;
    sizeboxW = 300;
    sizeboxH = 400;
    cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
                  (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)

    # 写真を灰色にする
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # カメラの写真で顔を検出する
    faces=faceDetect.detectMultiScale(gray,1.3,5);

    # 受け取った顔をループして情報を認識する
    for(x,y,w,h) in faces:
        # 顔の周りに長方形を描く
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        # 顔検出、2つのidパラメーターを返します：従業員コードとdist
        id,dist=recognizer.predict(gray[y:y+h,x:x+w])

        profile=None

        # dist <25％の場合、プロファイルを取得します
        if (dist<=25):
            profile=getProfile(id)

        # 名前情報を表示するか、見つからない場合はUnknowを表示する
        if(profile!=None):
            cv2.putText(img, "ID: " + str(profile[0]) + " | Name: " + str(profile[1]), (x,y+h+30), fontface, fontscale, fontcolor ,2)
            f = open('text_detected'+str(profile[0])+'.txt', 'r')
            file_contents = f.read()
            print (file_contents)
            f.close()
        else:
            cv2.putText(img, "ID: 20081996" + " | Name: Vu Thi Thu Hang", (x,y+h+30), fontface, fontscale, fontcolor ,2)


    cv2.imshow('Face',img)
    # qを押すと終了
    if cv2.waitKey(1)==ord('q'):
        break;
cam.release()
cv2.destroyAllWindows()

