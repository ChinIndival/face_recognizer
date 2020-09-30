
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('./test.jpg')

# Convert the image to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Make a copy of the original image to draw face detections on
image_copy = np.copy(image)

# Convert the image to gray 
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Detect faces in the image using pre-trained face dectector
faces = face_classifier.detectMultiScale(gray_image, 1.25, 6)

# Print number of faces found
print('Number of faces detected:', len(faces))

# Get the bounding box for each detected face
face_crop = []
for f in faces:
    x, y, w, h = [ v for v in f ]
    cv2.rectangle(image_copy, (x,y), (x+w, y+h), (255,0,0), 3)
    # Define the region of interest in the image  
    face_crop.append(gray_image[y:y+h, x:x+w])

for face in face_crop:
    cv2.imshow('face',face)
    cv2.waitKey(0)

cv2.imshow('face',face)
while True:

    try:
        check, frame = webcam.read()
        print(check)  # prints true as long as the webcam is running
        print(frame)  # prints matrix values of each framecd
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(filename='images.jpg', img=frame)
            r = random.randint(1, 20000000)
            img_file = 'images' + str(r) + '.jpg'
            cv2.imwrite(filename='data/' + img_file, img=frame)
            webcam.release()
            print("Processing image...")
            img_ = cv2.imread('images.jpg', cv2.IMREAD_ANYCOLOR)
            print("Image saved!")
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
