import cv2
import numpy as np
import vehicles  # our file
import time

cnt_up = 0
cnt_down = 0
cnt_right = 0
cnt_left = 0

cap = cv2.VideoCapture("cars.mp4")  # surveillance.m4v

# Get width and height of video

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # taking defult
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frameArea = h * w
areaTH = frameArea / 400

# Lines
line_up = int(2 * (h / 5))  # 2nd our of 4 lines
line_down = int(3 * (h / 5))  # 3rd our of 4 lines

up_limit = int(1 * (h / 5))  # 1st our of 4 lines
down_limit = int(4 * (h / 5))  # last our of 4 lines

print("Red line y:", str(line_down))
print("Blue line y:", str(line_up))
line_down_color = (255, 0, 0)
line_up_color = (255, 0, 255)
pt1 = [0, line_down]
pt2 = [w, line_down]
pts_L1 = np.array([pt1, pt2], np.int32)
pts_L1 = pts_L1.reshape((-1, 1, 2))
pt3 = [0, line_up]
pt4 = [w, line_up]
pts_L2 = np.array([pt3, pt4], np.int32)
pts_L2 = pts_L2.reshape((-1, 1, 2))

pt5 = [0, up_limit]
pt6 = [w, up_limit]
pts_L3 = np.array([pt5, pt6], np.int32)
pts_L3 = pts_L3.reshape((-1, 1, 2))
pt7 = [0, down_limit]
pt8 = [w, down_limit]
pts_L4 = np.array([pt7, pt8], np.int32)
pts_L4 = pts_L4.reshape((-1, 1, 2))

line_left = int(2 * (w / 5))  # 2nd our of 4 lines
line_right = int(3 * (w / 5))  # 3rd our of 4 lines

left_limit = int(1 * (w / 5))  # 1st our of 4 lines
right_limit = int(4 * (w / 5))  # last our of 4 lines

line_right_color = (255, 10, 10)
line_left_color = (255, 10, 155)

pt11 = [line_right, 0]
pt22 = [line_right, h]
pts_L11 = np.array([pt11, pt22], np.int32)
pts_L11 = pts_L11.reshape((-1, 1, 2))
pt33 = [line_left, 0]
pt44 = [line_left, h]
pts_L22 = np.array([pt33, pt44], np.int32)
pts_L22 = pts_L22.reshape((-1, 1, 2))

pt55 = [left_limit, 0]
pt66 = [left_limit, h]
pts_L33 = np.array([pt55, pt66], np.int32)
pts_L33 = pts_L33.reshape((-1, 1, 2))
pt77 = [right_limit, 0]
pt88 = [right_limit, h]
pts_L44 = np.array([pt77, pt88], np.int32)
pts_L44 = pts_L44.reshape((-1, 1, 2))

# Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Kernals
kernalOp = np.ones((4, 4), np.uint8)
kernalOp2 = np.ones((5, 5), np.uint8)
kernalCl = np.ones((11, 11), np.uint8)  # unit to unit8

font = cv2.FONT_HERSHEY_SIMPLEX
cars = []
max_p_age = 5
pid = 1

while (cap.isOpened()):  # to capture the frame of the video
    ret, frame = cap.read()
    for i in cars:
        i.age_one()
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)

    if ret == True:

        # Binarization
        ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        ret, imBin2 = cv2.threshold(fgmask2, 200, 255, cv2.THRESH_BINARY)
        # OPening i.e First Erode the dilate
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
        mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_CLOSE, kernalOp)

        # Closing i.e First Dilate then Erode
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernalCl)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernalCl)

        # Find Contours
        countours0, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # remove -,
        for cnt in countours0:
            area = cv2.contourArea(cnt)
            print(area)
            if area > areaTH:
                ####Tracking######
                m = cv2.moments(cnt)
                cx = int(m['m10'] / m['m00'])
                cy = int(m['m01'] / m['m00'])
                x, y, w, h = cv2.boundingRect(cnt)

                if cnt_left == 0 or cnt_right == 0:
                    new = True
                    if cy in range(up_limit, down_limit):
                        for i in cars:
                            if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:

                                new = False
                                i.updateCoords(cx, cy)

                                if i.going_UP(line_down, line_up) == True:
                                    cnt_up += 1
                                    print("ID:", i.getId(), 'crossed going up at', time.strftime("%c"))
                                elif i.going_DOWN(line_down, line_up) == True:
                                    cnt_down += 1
                                    print("ID:", i.getId(), 'crossed going up at', time.strftime("%c"))
                                break

                            if i.getState() == '1':
                                if i.getDir() == 'down' and i.getY() > down_limit:
                                    i.setDone()
                                elif i.getDir() == 'up' and i.getY() < up_limit:
                                    i.setDone()

                            if i.timedOut():
                                index = cars.index(i)
                                cars.pop(index)
                                del i

                        if new == True:  # If nothing is detected,create new
                            p = vehicles.Car(pid, cx, cy, max_p_age)
                            cars.append(p)
                            pid += 1

                if cnt_up == 0 and cnt_down == 0:
                    new = True
                    if cx in range(left_limit, right_limit):
                        for i in cars:
                            if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                                new = False
                                i.updateCoords(cx, cy)

                                if i.going_LEFT(line_right, line_left) == True:
                                    cnt_left += 1
                                    # print("left",cnt_left)
                                    print("ID:", i.getId(), 'crossed going left at', time.strftime("%c"))
                                elif i.going_RIGHT(line_right, line_left) == True:
                                    cnt_right += 1
                                    # print("left",cnt_right)
                                    print("ID:", i.getId(), 'crossed going right at', time.strftime("%c"))
                                break
                            if i.getState() == '1':
                                if i.getDir() == 'right' and i.getY() > right_limit:
                                    i.setDone()
                                elif i.getDir() == 'left' and i.getY() < left_limit:
                                    i.setDone()
                            if i.timedOut():
                                index = cars.index(i)
                                cars.pop(index)
                                del i

                        if new == True:  # If nothing is detected,create new
                            p = vehicles.Car(pid, cx, cy, max_p_age)
                            cars.append(p)
                            pid += 1

                # print("ansssss :::: ",str(cnt_right))
                # print("ansssss :::: ",str(cnt_left))
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for i in cars:
            cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), font, 0.3, i.getRGB(), 1, cv2.LINE_AA)

        str_down = 'DOWN: ' + str(cnt_down)
        str_up = 'UP: ' + str(cnt_up)
        str_right = 'RIGHT: ' + str(cnt_right)
        str_left = 'LEFT: ' + str(cnt_left)

        frame = cv2.polylines(frame, [pts_L1], False, line_down_color, thickness=2)
        frame = cv2.polylines(frame, [pts_L2], False, line_up_color, thickness=2)
        frame = cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
        frame = cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)

        frame = cv2.polylines(frame, [pts_L11], False, line_right_color, thickness=2)
        frame = cv2.polylines(frame, [pts_L22], False, line_left_color, thickness=2)
        frame = cv2.polylines(frame, [pts_L33], False, (255, 255, 255), thickness=1)
        frame = cv2.polylines(frame, [pts_L44], False, (255, 255, 255), thickness=1)

        cv2.putText(frame, str_up, (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, str_left, (100, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_left, (100, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str_right, (100, 90), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_right, (100, 90), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(10) & 0xff == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()









