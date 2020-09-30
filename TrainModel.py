import cv2,os
import numpy as np
import image
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

print("もう一度学習しています!")
def getImagesAndLabels(path):
    # ディレクトリ内のすべてのファイルを取得する
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #空の顔リストを作成する
    faceSamples=[]
    #空のIDリストを作成する
    Ids=[]
    #すべての画像パスをループし、IDと画像を読み込む
    for imagePath in imagePaths:
        if (imagePath[-3:]=="jpg"):
            #print(imagePath[0:])
            #画像を読み込んでグレースケールに変換する
            pilImage=Image.open(imagePath).convert('L')
            #PIL画像をnumpy配列に変換しています
            imageNp=np.array(pilImage,'uint8')
            #画像からIDを取得する
            Id=int(os.path.split(imagePath)[-1].split(".")[1])
            # トレーニング画像サンプルから顔を抽出する
            faces=detector.detectMultiScale(imageNp)
            #顔がある場合は、リストとそのIDに追加します
            for (x,y,w,h) in faces:
                faceSamples.append(imageNp[y:y+h,x:x+w])
                Ids.append(Id)
    return faceSamples,Ids


# dataSetディレクトリから顔とIDを取得する
faceSamples,Ids = getImagesAndLabels('dataSet')

# モデルをトレーニングして顔を特徴付​​け、各nahanペレットに割り当てます
recognizer.train(faceSamples, np.array(Ids))

# モデルを保存する
recognizer.save('recognizer/trainner.yml')

print("学習しました!")

