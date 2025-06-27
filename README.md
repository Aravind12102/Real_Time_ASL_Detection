# Real-Time ASL Sign Language Recognition

This project is a real-time hand gesture recognition system that identifies American Sign Language (ASL) alphabet letters using webcam input. It uses **MediaPipe** for landmark detection and a **K-Nearest Neighbors (KNN)** model trained on extracted hand landmarks.

---

## Features

- Real-time recognition of static ASL alphabet (A–Z) using a webcam.
- Hand tracking using MediaPipe’s robust hand landmark model.
- Achieves ~97% accuracy on the ASL alphabet dataset.
- Visual display of predicted letters with hand landmarks.
- Lightweight and easy to run on standard laptops.

---

##How to run the model

# 1st Method
- Clone the repo
- Run the following codes(can be performed both in jupyterNotebook or customIDEs)
  imports.py
  landmarkTraining.py
  KNN model.py
- Finally you will get a KNN pickle file
- Using this pickle file run the final python script "realtime.py" to enable realtime ASL recognition.

# 2nd Method
- The pickle files are already available in this repo
- Just import the pickle file and repeat the last step of 1st method

# A Sample Demo Video has been included to showcase the working model of the above project
  


