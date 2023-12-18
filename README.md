The cam_face srcipt uses opencv and DeepFace. The DeepFace analyzes the persons face and guesses their age and gender.

The code is setup so it works on the first face the camera sees. The code uses the variable analysis_done which is set to False outside of the while loop.
This ensures the DeepFace.analyze runs only once when running the script to reduce lag in the camera.
The only problem with this is that it only works on the first face and will apply the same label (age and gender) to any face it detects later.

I also added a cam_face_lag script which does the same thing as the cam_face script but it uses DeepFace analysis constantly, i.e. it doesn't have the analysis_done variable.
This provides dynamic results but causes the video to lag.

I added a img_face script that analyzes images instead of live video.
