import mediapipe as mp
import cv2
import numpy as np
import time
from pynput import keyboard
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

global GR
global KeyPresed
kb=keyboard.Controller()

class KeyInputs:
    KeyPresed=False
    KeyCount=int
    KeyDelay=10
    ALtTab=[kb._Key.alt, kb._Key.tab]
    UpArrow=[kb._Key.up]
    DownArrow=[kb._Key.down]
    WinMin=[kb._Key.cmd, kb._Key.down]
    # classes=['One','Peace','Three','Four','Palm','Fist','none']
    # classez=['One','Peace','Three','Four','Palm']
    classes=['One','Peace','Three','Four','Palm','Fist','none']
    classez=['One','Peace','Three','Four','Palm']

    def Kinput(self,j):
            if j=='Three':
                # for i in KeyInputs.WinMin:
                    # kb.press(i)
                # for i in KeyInputs.WinMin:
                    # kb.release(i)
                #KeyInputs.KeyPresed=True
                
            # elif j=='Four':
                # for i in KeyInputs.ALtTab:
                    # kb.press(i)
                # for i in KeyInputs.ALtTab:
                    # kb.release(i)
                
                KeyInputs.KeyPresed=True
            elif j=='Palm':
                kb.press(kb._Key.media_play_pause)
                #kb.release(kb._Key.media_play_pause)
                #KeyInputs.KeyPresed=True

            elif j=='One':
                kb.press(kb._Key.left)
                kb.release(kb._Key.left)
                #KeyInputs.KeyPresed=True

            elif j=='Peace':
                kb.press(kb._Key.right)
                kb.release(kb._Key.right)
                #time.sleep(3)
                KeyInputs.KeyPresed=True

def ResListener(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    for gesture in result.gestures:
        GR.append([category.category_name for category in gesture])
        print(GR)
        
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='Waive_v0.1.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=ResListener)

cap = cv2.VideoCapture(0)
results=[]
mpDraw = mp.solutions.drawing_utils
kyn=KeyInputs()
kyn.KeyCount=0
with GestureRecognizer.create_from_options(options) as recognizer:
    mpHands = mp.solutions.hands
    hands=mpHands.Hands()
    rGR=''
    
    while True:
        GR=[]
        ret, frame = cap.read()
        if not ret:
            break

        x, y, c = frame.shape
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results=hands.process(framergb)
        
        npcv=np.array(framergb)
        
        if results.multi_hand_landmarks:
            rGR=''
            landmarks = []
            for handslms in results.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS,
                                      mpDraw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mpDraw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
                
                #Recognize func requires time_stamp and image in an array form
                timestamp_ms = int(time.perf_counter() * 1000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=npcv)
                
                recognizer.recognize_async(mp_image,timestamp_ms)
                if GR:
                    rGR=(GR[0][0])
                #assign rGR with gesture name
                if kyn.KeyPresed==False:
                    if rGR in kyn.classez:
                        kyn.Kinput(rGR)
                        kyn.KeyPresed=True
        print(kyn.KeyPresed)
        print(rGR)
        if kyn.KeyPresed:
            kyn.KeyCount+=1
            if kyn.KeyCount>kyn.KeyDelay:
                kyn.KeyPresed=False
                kyn.KeyCount=0
                
        
        cv2.putText(frame, rGR, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 22, 250), 2, cv2.LINE_AA)
        cv2.imshow("Waive", frame)
        if cv2.waitKey(1) == ord('q'):
            break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()





# ''' here i encountered some errors but i found the code below from ai and im trying it out. the error---> (self._runner = _TaskRunner.create(graph_config, packet_callback)
                #    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# RuntimeError: Unable to open file at C:\Users\AbdulBari\.vscode\WorkPlace\Waive_v0.1.task )
# '''

# Open the model file in binary mode
model_file = open('Waive_v0.1.task', 'rb')

# Read the entire file
model_data = model_file.read()

# Close the file
model_file.close()

# Create the base options with the model data
base_options = BaseOptions(model_asset_buffer=model_data)

# Create the GestureRecognizer options
options = GestureRecognizerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=ResListener)



#for quiting the running code
