import face_recognition
import cv2
from picamera2 import Picamera2
import time
import pickle
import numpy as np
import serial
ser = serial.Serial('/dev/ttyUSB0',115200, timeout = 1.0)
time.sleep(3)
ser.reset_input_buffer()
print("Serial OK")


# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

cv_scaler = 1 # this has to be a whole number

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
rec = 0
#start_time = time.time()
#fps = 0

# --- Camera ---
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)})
)
picam2.start()

# --- Haar detector ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

authorized_names = ["john", "alice", "bob"]  # Replace with names you wish to authorise THIS IS CASE-SENSITIVE


def process_frame(frame):
    global face_locations, face_encodings, face_names
    
    # Resize the frame using cv_scaler to increase performance (less pixels processed, less time spent)
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    # Convert the image from BGR to RGB colour space, the facial recognition library uses RGB, OpenCV uses BGR
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    
    face_names = []
    authorized_face_detected = False
    
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        print(name)
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            print(name)
            # Check if the detected face is in our authorized list
            if name in authorized_names:
                authorized_face_detected = True

        #face_names.append(name)
    
    # Control the GPIO pin based on face detection
    #if authorized_face_detected:
        #output.on()  # Turn on Pin
   # else:
   #     output.off()  # Turn off Pin
    print("frame processed")
#     print(name)

    return frame

def draw_results(frame):
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left -3, top - 35), (right+3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
        # Add an indicator if the person is authorized
        if name in authorized_names:
            cv2.putText(frame, "Authorized", (left + 6, bottom + 23), font, 0.6, (0, 255, 0), 1)
    print("box drawn")
        #print(name)
    return frame

def detect_face_bbox(gray):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    # pick largest face (more stable)
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    #x, y, w, h = faces[0]
    return (int(x), int(y), int(w), int(h))

def create_mosse():
    #if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
      return cv2.legacy.TrackerMOSSE_create()
#     return cv2.TrackerMOSSE_create() 

def sendCoordinates(x,y,w,h):
    x2 = x + (w / 2)
    y2 = y + (h / 2)
    ser.write(str(x2).encode('utf-8') + b'\n')
    ser.write(str(y2).encode('utf-8') + b'\n')
    print("Sent Coordinates: ", x2, y2)
    #time.sleep(1)
    return (int(x2), int(y2))

tracker = None
bbox = None
tracking = False

while True:
    # Picamera2 returns RGB; convert once for OpenCV trackers/drawing
    frame_rgb = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    if not tracking:
        # Try to (re)detect
        found = detect_face_bbox(frame_gray)
        if found is not None:
            print(rec)
            if rec == 0:
                
                print("detected")
                process_frame(frame_rgb)
                
                draw_results(frame_rgb)
                
                #time.sleep(3)
                rec = 1 
                print(rec)
            bbox = found
            tracker = create_mosse()
            # IMPORTANT: init with the same 3-channel image type you'll use for update()
            tracker.init(frame_bgr, bbox)
            tracking = True
            # draw initial box for feedback
            x, y, w, h = bbox
            centerCoordinates = sendCoordinates(x,y,w,h)
            #cv2.circle(x2,y2,4, (0,0,255), -1)
            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.circle(frame_rgb, centerCoordinates, 4, (0,0,255), -1)
            cv2.putText(frame_rgb, "Initialized MOSSE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        else:
            cv2.putText(frame_rgb, "Searching...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        # Already tracking -> just update
        ok, new_box = tracker.update(frame_bgr)
        if ok:
            x, y, w, h = map(int, new_box)
            bbox = (x, y, w, h)
            centerCoordinates = sendCoordinates(x,y,w,h)
            #cv2.circle(x2,y2,4, (0,0,255), -1)
            cv2.circle(frame_rgb, centerCoordinates, 4, (0,0,255), -1)

            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame_rgb, "Tracking", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            # Lost -> go back to detection next loop
            tracking = False
            tracker = None
            cv2.putText(frame_rgb, "Lost — re-scanning", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("MOSSE Tracking (PiCam2)", frame_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop() 
cv2.destroyAllWindows()


