# Face-Tracking-Sentry
An automated sentry using a Raspberry Pi, ESP32, and two 9g servo motors along with a PiCamera 2 to perform facial detection, recognition, and tracking.
The control is split into two parts, image processing which is handled on the Raspberry Pi, and motor control which is handled by the ESP32-C3.
Due to the Raspberry Pi 4's relatively low processing power for tasks such as facial recognition, a finite state machine system is utilized to mitigate these issues.
The program initially starts in the detect state, which runs a continuous loop on standby until HaarCascades detects a face in the camera view. 
Once detection is achieved, the program then goes into the recognition state, this utilizes the facial recognition library to check the detected face against a file of pictures of labeled faces in the Pi's storage.
When recognition is completed an output of either the recognized faces name will be printed on the console, if the face is not recognized, "Unknown" will be printed on the console.
After this stage is completed, the system now goes to the Tracking state, this takes the same bounding box from the detection state and applies it as a boundary for a MOSSE tracker. 
The MOSSE tracker allows us to track a subjects face while also not losing our bounding box due to head movements such as turning the head to the side, up, or down.
The system will continue to stay in tracking mode until tracking is lost, which will cause the state machine to restart.
During the detection and tracking states, the coordinates of the center poitn of the bounding box are sent via USB serial communication to the EPS32.
These coordinates are used to calculate the error the bounding box to the center of the camera view.
I used this calculated error to control two servo motors in a PID control loop, with a seperate error calculation for the x and y axes. 
This allows the camera to have a quicker and more accurate response to the subject in the camera view. 
