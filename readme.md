# HAND GESTURE RECOGNITION
# overview
    This project aims to recognize hand gestures in real-time using openCV and HandDetectingModule for hand detection and a pre-trained Keras model for gesture classification. The system captures video from the webcam,detects hands and classifies the gestures using a CNN model trained on the Google teachable machine platform.
# File Structure
- 'training.py': the python script for training the model and to save the images in the given folder
- 'testing.py' : the python script for real-time hand gesture recogniztion
- 'Data/' : Directory to store the captured hand images for training.
- 'readme.md' : this file provides the information about the project.    
# Features
- Real-time hand gesture recognition.
- Supports multiple gestures with high accuracy.
- Easy to use 
- Customizable and extendable for different gestures.

# Requirements
- Python
- openCV
- cvzone
- Keras
- Tensorflow

# Usage 
Install all the dependencies before using the interface. The successfull usage of this repository goes:
1. clone the repository
2. Navigate to the project directory
3. Run the training.py
-  this script allows the user to add different gestures and also allows the users to save the cropped hand images by pressing the "s" key.
4. Run the testing.py
- this script detects the hands using the HandTrackingModule and performs the classification on the detected hand gestures using a pre-trained model loaded with classification module.
5. view the real-time hand gesture recognition on the screen.

# Model Training
The gesture classification model used in this project was trained using the Google Teachable Machine platform. The training process involves the follwoing steps:
1. Collect hand gesture images for each class you want to recognize .
2. Upload the images to the Google Teachable Machine Platform.
3. Train the model using the uploaded images.
4. Export the trained model along with a test file containing class labels.

# Credits
- HandTrackingModule by 'cvzone': https://github.com/cvzone/cvzone
- ClassifictaionModule by 'cvzone': https://github.com/cvzone/cvzone
- Gesture classification model trained with Google teachable Machine: https://teachablemachine.withgoogle.com/
- Course by : https://github.com/cvzone/cvzone





