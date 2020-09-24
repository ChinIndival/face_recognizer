import cv2
import sqlite3

cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 従業員情報をデータベースに追加する
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
    
id=input('従業員コードを入力してください:')
name=input('従業員の名前を入力してください: ')
print("スタッフの写真の撮影を開始し、qを押して終了します！")

insertOrUpdate(id,name)

sampleNum=0

while(True):

    ret, img = cam.read()

    # 画像を上下逆にします
    img = cv2.flip(img,1)

    # フレームを描く
    centerH = img.shape[0] // 2;
    centerW = img.shape[1] // 2;
    sizeboxW = 300;
    sizeboxH = 400;
    cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
                  (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)

    # 写真を灰色にする
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 顔認識
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # 受け取った顔の周りに長方形を描く
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        sampleNum = sampleNum + 1
        # 顔データをdataSetディレクトリに書き込みます
        cv2.imwrite("dataSet/User." + id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])

    cv2.imshow('frame', img)
    # qまたは 200以上のサンプル画像を押して終了
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break
    elif sampleNum>200:
        break

cam.release()
cv2.destroyAllWindows()
