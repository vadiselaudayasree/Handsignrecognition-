# Handsignrecognition-
This project is about recognizing hand signs in real-time using a webcam and a deep learning model. It detects American Sign Language (ASL) letters (except J and Z) and shows them as text on the screen.

🔍 What It Does
Captures live video from a webcam.
Detects the hand using MediaPipe.
Picks the hand area (Region of Interest).
Prepares the image for prediction.
Uses a CNN model to predict the hand sign.
Shows the letter on the screen and forms a sentence.
🧰 Tools and Technologies
Python
OpenCV – For capturing webcam video.
MediaPipe – For detecting hand landmarks.
TensorFlow / Keras – For training and using the deep learning model.
CNN (Convolutional Neural Network) – For recognizing hand signs from images.
📐 How It Works
Capture a frame from the webcam.
Detect hand landmarks using MediaPipe.
Extract and resize the hand image to 64x64 pixels.
Use the CNN model to predict the sign.
Show the result in real-time.
🧠 Model Info
Input: 64x64 RGB image

Output: 24 letters (A–Y, except J and Z)

Trained on a dataset of ASL hand signs

📂 Dataset
The dataset used for training the CNN model can be downloaded from the link below:

🔗 Download Dataset (Kaggle)

📸 Example Output
The system can detect and form words like “CALL” by recognizing each letter one by one from hand gestures.

Screenshot 2025-04-12 114833

⚠️ Problems Faced
In the beginning, the model was not accurate because of a small dataset.
The dataset did not have letters J and Z because they involve motion.
✅ Improvements Made
Used a bigger and better dataset with more variety.

Improved the model’s accuracy and performance.

📸 Demo
Below is an example prediction of the word "CALL" formed by recognizing individual hand signs:


