import cv2
import pathlib
from deepface import DeepFace

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))

images = ['Aleksandra_Prijovic.jpg', 'Ivan_Pernar.jpg', 'Johnny_Depp.jpg', 'Obama.jpg', 'Ryan_Reynolds.jpg', 'Zeljko_Pervan.jpeg']

for image in images:

    img = cv2.imread(f'images/{image}')
    result = DeepFace.analyze(img, actions=('age', 'gender'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = clf.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Age:{result[0]['age']}/Gender:{result[0]['dominant_gender']}"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    cv2.imshow('Face recognition', img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()





