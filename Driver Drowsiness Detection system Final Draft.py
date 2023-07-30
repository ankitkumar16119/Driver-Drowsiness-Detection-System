#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import dlib
import time
from tkinter import *
import threading
from PIL import Image, ImageTk
import pygame
from pygame import mixer
mixer.init(44100)
win = Tk()
win.title("Driver Drowsiness System: Online")
win.resizable(width=False, height=False)
win.geometry("700x750")
black = "#000000"
white = "#ffffff"
win.configure(background= black)
status = "normal"
# Define the eye aspect ratio
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Defining the threshold and consecutive frames required for drowsiness alert here
ear_threshold = 0.3
consecutive_frames = 48

# Initializing the frame counters
frame_counter = 0
alarm_status = False
key = cv2.waitKey(1) & 0xFF
#stop_flag = False

# Defining a function to play the alarm sound
def play_alarm_sound():
    pygame.mixer.music.load("Alarm.wav")
    pygame.mixer.music.play()
    
# Defining the number of frames to skip between face detection
detect_frequency = 5
detect_counter = 0

# Defining the scale factor for resizing the frame
scale_factor = 0.5

# Initializing the dlib face detector and landmark predictor data file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initializing user profile 
name = input("Enter driver name: ")
age = input("Enter Your Age: ")
history_file = "drowsiness_history.txt"

# Writing the driver profile to a text file
driver_profile = f"{name}, {age}"
with open("driver_profile.txt", "w") as f:
    f.write(driver_profile)

def write_history(driver_profile, drowsiness_detected):
    with open(history_file, "a") as f:
        if drowsiness_detected:
            f.write(f"{driver_profile} - Drowsiness detected\n")
        else:
            f.write(f"{driver_profile} - No drowsiness detected\n")

            
def MainCode():
    win.destroy()
    
# Starting the video stream capture
cap = cv2.VideoCapture(0)

#Creating the Graphical User Interface for the system
Header_label=Label(win, text="Driver Drowsiness Detection System", 
                   width = 100, height = 1, 
                   font = ('Calligrapher', '18', 'bold', 'underline'), 
                   bg = black, fg = white, anchor = "nw" )

Header_label.place(x = 150, y = 20)

name_label=Label(win, text="Name:", 
                 width = 5, height = 1, 
                 font = ('Comic Sans MS', '15'), 
                 bg = black, fg = white, anchor = "nw" )

name_label.place(x = 10, y = 60)

var_name_label=Label(win, text=name, 
                     width = 60, height = 1, 
                     font = ('Comic Sans MS', '15'), 
                     bg = black, fg = white, anchor = "nw" )

var_name_label.place(x = 90, y = 60)

age_label=Label(win, text="Age:", 
                width = 5, height = 1, 
                font = ('Comic Sans MS', '15'), 
                bg = black, fg = white, anchor = "nw" )

age_label.place(x = 10, y = 120)

var_age_label=Label(win, text=age, 
                    width = 60, height = 1, 
                    font = ('Comic Sans MS', '15'), 
                    bg = black, fg = white, anchor = "nw" )

var_age_label.place(x = 90, y = 120)

start_button = Button(win, text="Start", 
                      width=5, height=1, 
                      font = ('Comic Sans MS', '15','bold'), 
                      bg = black, fg = white, anchor = "center", command = MainCode)

start_button.place(x=300, y=180)

cam_label = Label(win)
cam_label.place(x= 20, y = 240)

#Creating a function to link the system camera to the GUI
def update():
    cv2image= cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    cam_label.imgtk = imgtk
    cam_label.configure(image=imgtk)
    cam_label.after(20, update)
update()
    
win.mainloop()

# Starting the loop for capturing frames
while True:
    # Reading frames from the video capture
    ret, frame = cap.read()

    # Resizing the frames for processing
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

    # Converting the frames to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if detect_counter == 0:
        # Detecting the faces in the grayscale frame
        faces = detector(gray)
        detect_counter = detect_frequency
    else:
        detect_counter -= 1

    # Loop through each face detected
    for face in faces:
        # Detecting the landmarks of the face
        landmarks = predictor(gray, face)

        # Extracting the left and right eye landmarks
        left_eye = []
        right_eye = []
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))

        # Calculations for the eye aspect ratio for each eye
        left_ear = eye_aspect_ratio(np.array(left_eye))
        right_ear = eye_aspect_ratio(np.array(right_eye))

        # Taking average of the eye aspect ratios
        ear = (left_ear + right_ear) / 2.0

        # If the eye aspect ratio is below the threshold, increment the frame counter
        if ear < ear_threshold:
            frame_counter += 1

            # If the eyes have been below the threshold for a certain amount of time, sound the alarm
            if frame_counter >= consecutive_frames:
                if not alarm_status:
                    alarm_status = True
                    print("Drowsiness detected! Press Q to Exit the System!!")
                    #status = "Drowsy"
                    # Play the alarm sound in a separate thread
                    threading.Thread(target=play_alarm_sound).start()
                    # Writing the driver's profile and detection result to the history file
                    write_history(driver_profile, True)
                                           
        else:
            frame_counter = 0
            alarm_status = False

        # Drawing the contours arounf the eye
        left_eye_hull = cv2.convexHull(np.array(left_eye))
        right_eye_hull = cv2.convexHull(np.array(right_eye))
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

    # Displaying system Cam
    cv2.imshow("Driver Drowsiness Detection System: Online", frame)



# Wait for a key press and quit if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        pygame.mixer.music.stop()
        # Write the driver's profile and detection result to the history file before quitting
        write_history(driver_profile, False)
        f.close()
        break 

# Release the video stream and destroy all windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




