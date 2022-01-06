from camera_config.camera import CameraStream, GstStream
import argparse

import cv2

camera1 = "rtsp://admin:NybSys123!@10.200.10.216/"
# camera1 = 'britom.mp4'

ap = argparse.ArgumentParser()
ap.add_argument("--width", "-w", type=int, default=800)
ap.add_argument("--height", "-t", type=int, default=600)
ap.add_argument("--source", "-s", type=str, default=camera1)

args = vars(ap.parse_args())

try:
    source = int(args['source'])
    source = CameraStream(src=source).start()
except:
    source = CameraStream(src=args['source']).start()


# while True:
#     text = 'No Motion'
#     ret, frame = source.read()
#     cv2.imshow("Security Feed", frame)
    
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break


# cv2.destroyAllWindows()