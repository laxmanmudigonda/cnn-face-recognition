import cv2
import numpy as np
import os
from keras.models import load_model

model = load_model("face_model.h5")

dataset = "dataset"

labels = [name for name in os.listdir(dataset) if os.path.isdir(os.path.join(dataset,name))]

cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:

    ret, frame = cam.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face = gray[y:y+h,x:x+w]
        face = cv2.resize(face,(100,100))

        face = face/255.0
        face = face.reshape(1,100,100,1)

        prediction = model.predict(face)

        label_index = np.argmax(prediction)

        name = labels[label_index]

        cv2.putText(frame,name,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Face Recognition",frame)

    if cv2.waitKey(1)==27:
        break

cam.release()
cv2.destroyAllWindows()