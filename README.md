The cam_face srcipt uses opencv and deepface. The deepface analyzes the persons face and determins their age and gender.

The code is setup so it works on the first face the camera sees. That is thanks to this part of the code:
    if not analysis_done:
        result = DeepFace.analyze(frame, actions=('age', 'gender'), enforce_detection=False)
        analysis_done = True
This part of the code uses the variable analysis_done wich is set to False outside of the while loop.
This ensures the DeepFace.analyze runs only once when running the script to reduce lag in the camera.

The only problem with this is that it only works on the first face and will apply the same label (age and gender) to any face it detects later.

The opencv part was used for creating rectangles around faces:
    for (x,y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"Age:{result[0]['age']}/Gender:{result[0]['dominant_gender']}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
I used the haarcascade_frontalface_default.xml file for opencv face detection.