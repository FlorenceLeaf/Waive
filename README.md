# Waive
Ai powered/trained hand recognitions software to execute custome commands set by user, different command per gesture

You need to have these installed:
mediapipe
opencv-python
numpy
pynput

You can install them using pip:
"pip install mediapipe opencv-python numpy pynput"

External Model File (Waive_v0.1.task)(Critical)
You must have this file present in the same directory where you run the Python script
This .task file is a custom MediaPipe model that the program loads.

If it’s missing, you will get this error:

RuntimeError: Unable to open file

For Hardware Requirment:
A Webcam
 the program is Pretty lightwaighted so interms of hardware requirment a webcam is needed since the program needs to "see" the users hands to execute commands

 theres a couple of commands commented out since they were not needed at the time but to enable them you just need to uncomment them

So to summarize, you need:

Requirement	Details
Python packages	mediapipe, opencv-python, numpy, pynput
Model file	Waive_v0.1.task placed correctly
Webcam	For hand tracking input



for ease i will in future create a requirment file that installs mediapipe, opencv etc

requirements.txt:
  mediapipe
  opencv-python
  numpy
  pynput

then, pip install -r requirements.txt


