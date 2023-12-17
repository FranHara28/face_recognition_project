import cv2
from deepface import DeepFace

camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()

    result = DeepFace.analyze(frame, actions=('age', 'gender'), enforce_detection=False)
    position_info = result[0]['region']

    cv2.rectangle(frame, (position_info['x'], position_info['y']), (position_info['x']+position_info['w'], position_info['y']+position_info['h']), (0, 255, 0), 2)
    label = f"Age:{result[0]['age']}/Gender:{result[0]['dominant_gender']}"
    cv2.putText(frame, label, (position_info['x'], position_info['y'] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
