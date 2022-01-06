import cv2
from scipy.linalg.special_matrices import tri
import dlib
from scipy.spatial import distance
from app import source
from gaze_tracking import GazeTracking
import time


################################
###Attention
################################
def attention(attention_level, trigger=None):
    if trigger == '+':
        attention_level +=1
    elif trigger == '-':
        if attention_level > 0:
            attention_level -= 1
        else:
            pass
    return attention_level

##############################
########Flag+Counter
##############################
def visibility_counter(FLAG_FACE_NOT, count_not_face_frame):
    if(FLAG_FACE_NOT == True):
        print("Sadness is eternal")
        print(count_not_face_frame)
        count_not_face_frame += 1
    elif(FLAG_FACE_NOT == False):
        count_not_face_frame = 0

    return count_not_face_frame

########################################
########Increase Brightness
########################################
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
    
###############################################
#####Equation
#####EAR=(||p1-p5||+||p2-p4||)/(2.0*||p0-p3||)
###############################################
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

##################################
####Convert frame to grayscale
##################################
def rbg_grayscale(frame):

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_frame

##################################
####Gaze prediction 
##################################
def gaze_prediction(frame, gaze):

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

        return text

##################################
####Frame Text Encode
##################################
def encode_gaze_text_frame(frame, text):
    cv2.putText(frame, text, (100, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 1)
    return frame

##################################
####Draw Eye (Landmarks)
##################################
def draw_eye_frame(starting_landmark, ending_landmark):
    ##################
    #########Eye Draw
    ##################
    Eye = []
    for n in range(starting_landmark,ending_landmark):
        x = face_landmarks.part(n).x
        y = face_landmarks.part(n).y
        Eye.append((x,y))
        next_point = n+1
        if n == (ending_landmark - 1):
            next_point = starting_landmark
        x2 = face_landmarks.part(next_point).x
        y2 = face_landmarks.part(next_point).y

        #Encode eye co-ordinates to image
        cv2.line(frame,(x,y),(x2,y2),(255,255,0),1)
    return frame, Eye

#########################
####Camera/Source
#########################
cap = cv2.VideoCapture(-1)
# cap = cv2.VideoCapture('sameer_mod.mp4')
# cap = cv2.VideoCapture('britom_mod.mp4')
# cap = source

#########################
####Initialize Detectors
#########################
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor('./gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat')

##################################
####Initialize necessary variables
##################################
attention_level = 0
count_not_face_frame = 0
FLAG_FACE_NOT = True

##################################
####Main Code
##################################
while True:
    #try:
        start = time.time()
        _, frame = cap.read()
        
        ##convert to grayscale
        gray_frame = rbg_grayscale(frame)

        ##increase brightness of rgb frame
        #frame = increase_brightness(frame, value=30)

        ##detected faces list from hog_face
        faces = hog_face_detector(gray_frame)
        
        ######################################################
        ##Third choice
        #######################################################
        print(faces)
        if(len(faces)==0):
            #reset counter if previously false
            count_not_face_frame = visibility_counter(FLAG_FACE_NOT, count_not_face_frame)
            
            ####
            FLAG_FACE_NOT=True
            print("ping")
            #print("1111111111111111",FLAG_FACE_NOT)
            
            if(len(faces)==0 and FLAG_FACE_NOT==True):
                print("Kill me")
                #increment counter as face not present
                count_not_face_frame = visibility_counter(FLAG_FACE_NOT, count_not_face_frame)
                #sanity check
                #print("2222222222222222222222",FLAG_FACE_NOT)
                #print("333333333333333333333333",count_not_face_frame)
                
                if(count_not_face_frame>200):
                    print("##############################Driver is gonna die, As individual's sleepy######################################")
                    #reset if 200
                    #reset counter as we have found that drive is dead
                    FLAG_FACE_NOT=False
                    count_not_face_frame = visibility_counter(FLAG_FACE_NOT, count_not_face_frame)
                    # cv2.imshow("Are you Sleepy", frame)
                cv2.imshow("Are you Sleepy", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
        
        #################################################
        #This branch of code only runs when face is found
        #################################################
        FLAG_FACE_NOT=False
        print("Driver is still alive but soon gonna dead")
        #######################################################

        
        for face in faces: 
            face_landmarks = dlib_facelandmark(gray_frame, face)


            #########Gaze Tracking
            gaze = GazeTracking(face, face_landmarks)

            #####################################################################
            ###NOTE:Gaze.refresh outside Function because of local scope error
            #####################################################################
            gaze.refresh(frame)
            #gaze predicting library, returns text
            text = gaze_prediction(frame, gaze)
            #encode gaze text
            encode_gaze_text_frame(frame, text)

            leftEye = []
            rightEye = []
            
            #Get Left and Right Eye Co-ordinates
            #Draw them on frame
            frame, leftEye = draw_eye_frame(36, 42)
            frame, rightEye = draw_eye_frame(42, 48)

            #Calculate Left and Right EAR
            left_ear = calculate_EAR(leftEye)
            right_ear = calculate_EAR(rightEye)

            #ROUND EAR
            EAR = (left_ear+right_ear)/2
            #EAR = round(EAR,2)
            print(EAR)
            
            if EAR<0.26:
                attention_level = attention(attention_level,'+')
            else:
                attention_level = attention(attention_level,'-')

            if attention_level > 10:
                cv2.putText(frame,"DROWSY",(20,100),
                    cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
                cv2.putText(frame,"Are you Sleepy?",(20,400),
                    cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
                print("##############################Drowsy####################################")
                print("##############################Drowsy####################################")
                print("##############################Drowsy####################################")
                #NOTE: NO COOLDOWN, To Reset Attention Level
                attention_level= 0

            # print(EAR)
            # print("Frames: ", count)
        print('Attention Level :',attention_level)
        end = time.time()
        fps = 1/(end-start)
        print("Frames: ", fps)
        cv2.imshow("Are you Sleepy", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    # except:
    #     print("##########################CODE HAS BEEN PASSED TO EXCEPT################################")
    #     pass
cv2.destroyAllWindows()