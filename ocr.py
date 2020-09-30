import cv2
from time import sleep
import requests
import io
import json
import os
import random
import numpy as np
import sqlite3

#情報をデータベースに追加する
def insertOrUpdate(id, name):
    conn=sqlite3.connect("FaceBaseNew.db")
    cursor=conn.execute('SELECT * FROM People WHERE ID='+str(id))
    isRecordExist=0
    for row in cursor:
        isRecordExist = 1
        break

    if isRecordExist==1:
        cmd="UPDATE people SET Name=' "+str(name)+" ' WHERE ID="+str(id)
    else:
        cmd="INSERT INTO people(ID,Name) Values("+str(id)+",' "+str(name)+" ' )"

    conn.execute(cmd)
    conn.commit()
    conn.close()

#IDとお名前を保存する  
id=input('IDを入力してください:')
name=input('お名前を入力してください: ')

#Insert, updateデータ
insertOrUpdate(id,name)

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
sleep(2)

print("証明書の写真を寄ってください")
while True:

    try:
        check, frame = webcam.read()
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(filename='id_card_images.jpg', img=frame)
            r = random.randint(1, 20000000)
            img_file = 'images' + str(r) + '.jpg'
            cv2.imwrite(filename='id_card/' + img_file, img=frame)
            webcam.release()
        
            img_ = cv2.imread('id_card_images.jpg', cv2.IMREAD_ANYCOLOR)
 
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

#sleep(2)

resim = "id_card_images.jpg"
img = cv2.imread(resim)
print("証明書を認識しました！")
api = img

# APIでOcrを実装する
url_api = "https://api.ocr.space/parse/image"
_, compressedimage = cv2.imencode(".jpg", api, [1, 90])
file_bytes = io.BytesIO(compressedimage)

result = requests.post(url_api,
                       files={resim: file_bytes},
                       data={"apikey": "helloworld",
                             "language": "eng"})

result = result.content.decode()
print(result)
result = json.loads(result)

parsed_results = result.get("ParsedResults")[0]
text_detected = parsed_results.get("ParsedText")
print(text_detected)

#Txtファイルに書く
print("ファイルに情報を書いています。。。")
f = open("text_detected"+str(id)+".txt", "a+")
f.write(text_detected)
f.close()
print("ファイルに情報を書きました！")

cv2.imshow("roi", api)
cv2.imshow("Img", img)
cv2.waitKey(0)

#証明書から写真を取る
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('./test3.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_copy = np.copy(image)
 
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

faces = face_classifier.detectMultiScale(gray_image, 1.25, 6)

print('顔：', len(faces))

sampleNum=0
# 顔を表示する
face_crop = []
for f in faces:
    x, y, w, h = [ v for v in f ]
    cv2.rectangle(image_copy, (x,y), (x+w, y+h), (255,0,0), 3)
    # Define the region of interest in the image  
    face_crop.append(gray_image[y:y+h, x:x+w])

for face in face_crop:
    cv2.imshow('face',face)
    cv2.imwrite("dataSet/User." + id + '.' + str(sampleNum) + ".jpg", face)
    cv2.waitKey(0)
