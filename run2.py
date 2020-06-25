import cv2
from tracker import KCFTracker
import numpy as np
import sys
import os
from yolo import YOLO
from time import time
from collections import deque
from keras import backend
from PIL import Image
from sympy import Symbol, Matrix
#import math

def main(yolo):
    selectingObject = False#use detect not hand
    initTracking = True
    onTracking = False
    duration = 0.01
    pts = deque(maxlen=50)
    bx2 = [0,0,0,0] 
    tracker = KCFTracker(True, True, True)
    #KF initialization
    P = np.diag([3000.0, 3000.0, 3000.0, 3000.0])
    I = np.eye(4)
    H = np.matrix([[0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
    ra = 0.15  # 厂商提供
    R = np.matrix([[ra, 0.0],
                   [0.0, ra]])
    noise_ax = 0.3
    noise_ay = 0.3
    cv2.namedWindow('tracking')
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('./oc1.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (selectingObject):
            continue
        elif (initTracking):
            image = Image.fromarray(frame[...,::-1]) #bgr to rgb
            box,class_names = yolo.detect_image(image)#use yolo to predict
            ix0=int(box[0][0])
            iy=int(box[0][1])
            w=int(box[0][2])
            h=int(box[0][3])
            tracker.init([ix0, iy, w, h], frame)
            #Lenovo camera
            ixx0=int((box[0][0])+(box[0][2])/2)
            iyy0=int((box[0][1])+(box[0][3])/2)
            
            #GH4 camera
            '''
            D0 = 2*( 355.2/ w)  # meter,qianhou
            x_cen = 640
            xd0 = ixx0 - x_cen
            # geo similarity
            X0 = D0 / 887.9 * xd0  # meter,zuoyou
            '''
            #xiaomi
            
            D0 = (1978.9/h)  # meter,qianhou
            x_cen = 703
            xd0 = ixx0 - x_cen
            # geo similarity
            X0 = xd0*(1.54/h)  # meter,zuoyou
            
            #humw.append(0)
            #humw.append(w)
            #DJI
            '''
            D0 = (3006.5/h)  # meter,qianhou
            x_cen = 1361.8
            xd0 = ixx0 - x_cen
            # geo similarity
            X0 = xd0*(1.75/h)  # meter,zuoyou
            '''
            state = np.matrix([[X0, D0, 0.0, 0.0]]).T
            state_2D = np.matrix([[ixx0, iyy0 , 0.0, 0.0]]).T
            initTracking = False
            onTracking = True
        elif (onTracking):
            t0 = time()
            boundingbox = tracker.update(frame)
            x=boundingbox[0]
            y=boundingbox[1]
            w=boundingbox[2]
            h=boundingbox[3]
            cx = int(x + w / 2)
            cy = int(y + h / 2)              
            boundingbox = list(map(int, boundingbox))
            x1=boundingbox[0]
            y1=boundingbox[1]
            w1=boundingbox[2]
            h1=boundingbox[3]
            ix=int((boundingbox[0])+(boundingbox[2])/2)
            iy=int((boundingbox[1])+(boundingbox[3])/2)
            ht=boundingbox[3]
            wt=boundingbox[2]
            #GH$
            '''
            D = 2*( 355.2/ wt)  # meter,qianhou
            x_cen = 640
            xd = ix - x_cen
            # geo similarity
            X = D / 887.9 * xd  # meter,zuoyou
            '''
            #xiaomi
            
            D = (1978.9/ht)  # meter,qianhou
            x_cen = 703
            xd0 = ix0 - x_cen
            # geo similarity
            X = xd0*(1.52/ht)  # meter,zuoyou
            
            '''
            D = (3006.5/ht)  # meter,qianhou
            x_cen = 1361.8
            xd0 = ix0 - x_cen
            # geo similarity
            X = xd0*(1.75/ht)
            '''
            #K = D/(D+60)
            
            #if ():
            #else:
               # bx2[0] = int(x+w1/20)
               # bx2[1] = int(y-h1/20)
                #bx2[2] = int(w)
                #bx2[3] = int(h)
            td=time()
            dt = td-t0
            #print(dt)  # Time step between Filters steps
            F = np.matrix([[1.0, 0.0, dt, 0.0],
                           [0.0, 1.0, 0.0, dt],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
            dt_2 = dt * dt
            dt_3 = dt_2 * dt
            dt_4 = dt_3 * dt
            Q = np.matrix([[0.25 * dt_4 * noise_ax, 0, 0.5 * dt_3 * noise_ax, 0],
                           [0, 0.25 * dt_4 * noise_ay, 0, 0.25 * dt_3 * noise_ay],
                           [dt_3 / 2 * noise_ax, 0, dt_2 * noise_ax, 0],
                           [0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay]])
            dX=X-state[0][0]
            dD=D-state[1][0]
            dVx=int(dX/dt)
            dVd=int(dD/dt)
            dX2=cx-state_2D[0][0]
            dD2=cy-state_2D[1][0]
            dVx2=int(dX2/dt)
            dVd2=int(dD2/dt)
            state = F * state  # Project the state ahead
            state_2D = F * state_2D
            P = F * P * F.T + Q  # Project the error covariance ahead
            # Measurement Update (Correction)
            # ==============================
            S = H * P * H.T + R
            K = (P * H.T) * np.linalg.pinv(S)
            # Update the estimate via z
            Z = ([[float('%.2f'% dVx)],[float('%.2f'% dVd)]])# 本身应该用速度传感器测量真值的，但是由于没有这个传感器所以只能通过视频中算出来的X，D除去对应的dt计算。
            Z2 = ([[float('%.2f'% dVx2)],[float('%.2f'% dVd2)]])
            ykf = Z - (H * state)
            y2kf = Z2 - (H*state_2D)
            state = state + (K * ykf)
            state_2D = state_2D + (K * y2kf)
            # update the error convariance
            P = (I - (K * H)) * P
            #draw the picture and give out the info of interested obj
            #print(str(int(state_2D[2][0])))
            #print(str(int(state_2D[3][0])))
            if (abs(int(state_2D[2][0]))-abs(int(state_2D[3][0]))> 29):
                if (int(state_2D[2][0])>0):
                    #right
                    bx2[0] = int(x + w / 15)
                    bx2[1] = int(y - h / 15)
                    bx2[2] = int(w)
                    bx2[3] = int(h)

                    cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
                    cv2.rectangle(frame, (bx2[0], bx2[1]), (bx2[0] + bx2[2], bx2[1] + bx2[3]), (0, 0, 255), 2)

                    cv2.line(frame, (x1, y1), (bx2[0], bx2[1]), (255, 0, 255), 2)
                    cv2.line(frame, (x1 + w1, y1), (bx2[0] + bx2[2], bx2[1]), (255, 0, 255), 2)
                    cv2.line(frame, (x1, y1 + h1), (bx2[0], bx2[1] + bx2[3]), (255, 0, 255), 2)
                    cv2.line(frame, (x1 + w1, y1 + h1), (bx2[0] + bx2[2], bx2[1] + bx2[3]), (255, 0, 255), 2)

                    cv2.line(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 255), 1)
                    cv2.line(frame, (bx2[0], bx2[1]), (bx2[0] + bx2[2], bx2[1] + bx2[3]), (100, 100, 50), 1)
                    cv2.line(frame, (x1 + w1, y1), (x1, y1 + h1), (255, 255, 0), 1)
                    cv2.line(frame, (bx2[0] + bx2[2], bx2[1]), (bx2[0], bx2[1] + bx2[3]), (100, 75, 50), 1)

                    cv2.arrowedLine(frame, (int((x1+bx2[0])/2 + w1), int((y1+bx2[1])/2+h1) ), (int((x1+bx2[0])/2 + w1+50), int((y1+bx2[1])/2+h1) ), (0, 0, 255), 3)



                else:
                    #left
                    bx2[0] = int(x + w / 15)
                    bx2[1] = int(y - h / 15)
                    bx2[2] = int(w)
                    bx2[3] = int(h)

                    cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
                    cv2.rectangle(frame, (bx2[0], bx2[1]), (bx2[0] + bx2[2], bx2[1] + bx2[3]), (0, 0, 255), 2)

                    cv2.line(frame, (x1, y1), (bx2[0], bx2[1]), (255, 0, 255), 2)
                    cv2.line(frame, (x1 + w1, y1), (bx2[0] + bx2[2], bx2[1]), (255, 0, 255), 2)
                    cv2.line(frame, (x1, y1 + h1), (bx2[0], bx2[1] + bx2[3]), (255, 0, 255), 2)
                    cv2.line(frame, (x1 + w1, y1 + h1), (bx2[0] + bx2[2], bx2[1] + bx2[3]), (255, 0, 255), 2)

                    cv2.line(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 255), 1)
                    cv2.line(frame, (bx2[0], bx2[1]), (bx2[0] + bx2[2], bx2[1] + bx2[3]), (100, 100, 50), 1)
                    cv2.line(frame, (x1 + w1, y1), (x1, y1 + h1), (255, 255, 0), 1)
                    cv2.line(frame, (bx2[0] + bx2[2], bx2[1]), (bx2[0], bx2[1] + bx2[3]), (100, 75, 50), 1)


                    
                    cv2.arrowedLine(frame, (int((x1+bx2[0])/2), int((y1+bx2[1])/2+h1)), (int((x1+bx2[0])/2)-50, int((y1+bx2[1])/2+h1)), (0, 0, 255), 3)


            else:
                if (int(state_2D[3][0])<0):
                    #back
                    bx2[0] = int(x + w / 5)
                    bx2[1] = int(y - h/ 5)
                    bx2[2] = int(w)
                    bx2[3] = int(h)

                    cv2.rectangle(frame, (x1, y1),(x1 + w1, y1 + h1), (0, 0, 255), 2)
                    cv2.rectangle(frame,(bx2[0],bx2[1]),(bx2[0] + bx2[2], bx2[1] + bx2[3]), (0, 0, 255), 2)

                    cv2.line(frame,(x1,y1),(bx2[0],bx2[1]),(255, 0, 255),2)
                    cv2.line(frame,(x1+w1,y1),(bx2[0]+bx2[2],bx2[1]), (255, 0, 255), 2)
                    cv2.line(frame,(x1,y1+ h1), (bx2[0],bx2[1]+bx2[3]), (255, 0, 255), 2)
                    cv2.line(frame,( x1+w1 ,y1+h1), (bx2[0]+bx2[2],bx2[1]+bx2[3]), (255, 0, 255), 2)

                    cv2.line(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 255), 1)
                    cv2.line(frame, (bx2[0],bx2[1]), (bx2[0] + bx2[2], bx2[1] + bx2[3]), (100, 100, 50), 1)
                    cv2.line(frame, (x1+w1, y1), (x1, y1+h1), (255, 255, 0), 1)
                    cv2.line(frame, (bx2[0]+bx2[2], bx2[1]), (bx2[0], bx2[1]+bx2[3]), (100, 75, 50), 1)

                    cv2.arrowedLine(frame, (int(bx2[0]+bx2[2]/2),bx2[1]+bx2[3]),(int(bx2[0]+bx2[2]/2+50),bx2[1]+bx2[3]-50) , (0, 0, 255),3)  



                else:
                    #front
                    bx2[0] = int(x+w/5)
                    bx2[1] = int(y-h/5)
                    bx2[2] = int(w)
                    bx2[3] = int(h)

                    cv2.rectangle(frame, (x1, y1),(x1 + w1, y1 + h1), (0, 0, 255), 2)
                    cv2.rectangle(frame,(bx2[0],bx2[1]),(bx2[0] + bx2[2], bx2[1] + bx2[3]), (0, 0, 255), 2)

                    cv2.line(frame,(x1,y1),(bx2[0],bx2[1]),(255, 0, 255),2)
                    cv2.line(frame,(x1+w1,y1),(bx2[0]+bx2[2],bx2[1]), (255, 0, 255), 2)
                    cv2.line(frame,(x1,y1+ h1), (bx2[0],bx2[1]+bx2[3]), (255, 0, 255), 2)
                    cv2.line(frame,( x1+w1 ,y1+h1), (bx2[0]+bx2[2],bx2[1]+bx2[3]), (255, 0, 255), 2)

                    cv2.line(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 255), 1)
                    cv2.line(frame, (bx2[0],bx2[1]), (bx2[0] + bx2[2], bx2[1] + bx2[3]), (100, 100, 50), 1)
                    cv2.line(frame, (x1+w1, y1), (x1, y1+h1), (255, 255, 0), 1)
                    cv2.line(frame, (bx2[0]+bx2[2], bx2[1]), (bx2[0], bx2[1]+bx2[3]), (100, 75, 50), 1)

                    cv2.arrowedLine(frame, (int(x1+w1/2),y1+h1),(int(x1+w1/2-50),y1+h1+50), (0, 0, 255), 3)                   


            center = (int(cx), int(cy))
            pts.appendleft(center)
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                cv2.line(frame, (pts[i - 1]), (pts[i]), (0,255,0), 2)
            t1=time()
            duration = 0.8 * duration + 0.2 * (t1 - t0)
            cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2)
            cv2.putText(frame, 'Pedestrian status camera-coor: X,D:' + str(state[0][0])+str(state[1][0]), (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2)
            cv2.putText(frame, 'Pedestrian status camera-coor: Vx,Vd:' + str(state[2][0])+str(state[3][0]), (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2)
            #print('2D parameter')
            #print(str(state_2D[2][0]))
            #print(str(state_2D[3][0]))
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(20) & 0xFF
        if c == 27 or c == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
