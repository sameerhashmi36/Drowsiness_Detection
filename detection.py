import cv2
import dlib
from scipy.spatial import distance
from app import source
from gaze_tracking import GazeTracking
import time


############################
#####Equation
#####EAR=(||p1-p5||+||p2-p4||)/(2.0*||p0-p3||)
############################
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

#########################
####Camera
#########################
cap = cv2.VideoCapture(-1)
# cap = cv2.VideoCapture('sameer_mod.mp4')
# cap = source

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor('./gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat')

count = 0
temp_list = []

##################################
####Main Code
##################################
while True:
    # try:
        if len(temp_list) >= 20:
            temp_list = temp_list[10:]
            print(temp_list)

        start = time.time()
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        faces = hog_face_detector(gray)
        for face in faces:

            face_landmarks = dlib_facelandmark(gray, face)

            ################################
            ######Gaze Tracking
            ################################
            gaze = GazeTracking(face, face_landmarks)
            gaze.refresh(frame)

            text = ''
            if gaze.is_right():
                text = 'Looking Right!!!!!'
                # print(text)
            elif gaze.is_left():
                text = 'Looking Left!!!!!'
                # print(text)
            elif gaze.is_center():
                text = 'Looking Front'
                # print(text)
            
            cv2.putText(frame, text, (100, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 1)

            leftEye = []
            rightEye = []

            for n in range(36,42):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                leftEye.append((x,y))
                next_point = n+1
                if n == 41:
                    next_point = 36
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame,(x,y),(x2,y2),(255,255,0),1)

            for n in range(42,48):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                rightEye.append((x,y))
                next_point = n+1
                if n == 47:
                    next_point = 42
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame,(x,y),(x2,y2),(255,255,0),1)

            left_ear = calculate_EAR(leftEye)
            right_ear = calculate_EAR(rightEye)

            EAR = (left_ear+right_ear)/2
            EAR = round(EAR,2)
            print(EAR)
            if EAR<0.26:
                temp_list.append(1)
                print("1111111111111",temp_list)
                #############################
                ########Sum
                #############################
                if sum(temp_list)>10:
                    cv2.putText(frame,"DROWSY",(20,100),
                        cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
                    cv2.putText(frame,"Are you Sleepy?",(20,400),
                        cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
                    print("Drowsy")
            else:
                temp_list.append(0)
                # print("00000000000",temp_list)

            # count += 1
            # print(EAR)
            # print("Frames: ", count)
        end = time.time()
        fps = 1/(end-start)
        print("Frames: ", fps)
        cv2.imshow("Are you Sleepy", frame)
        # cv2.imshow("gray", gray)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    # except:
    #     pass
cv2.destroyAllWindows()