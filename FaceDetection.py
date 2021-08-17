import cv2
import os

dataset = "dataset"
name = "Niharika"

path = os.path.join(dataset, name)
if not os.path.isdir(path):
    os.mkdir(path)

cam = cv2.VideoCapture(0)
alg = 'haarcascade_frontalface_default.xml'
haar_cascad = cv2.CascadeClassifier(alg)
count = 1
while count<31:
    _, img = cam.read()
    text = "Person not detected"
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face = haar_cascad.detectMultiScale(gray, 1.3, 4)
    for(x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        faceOnly = gray[y:y+h, x:x+w]
        resize_img = cv2.resize(faceOnly, (150,150))
        cv2.imwrite("%s/%s.jpg" %(path, count), resize_img)
        text = "Person Detected"
        count += 1
    print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("Face Detection", img)
    if cv2.waitKey(1) == 27:
        break
print("Image successfully Captured")
cam.release()
cv2.destroyAllWindows()
