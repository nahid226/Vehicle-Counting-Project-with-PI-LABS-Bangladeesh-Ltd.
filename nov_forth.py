import cv2
import time
import math
import datetime
import numpy as np
from PIL import ImageTk, Image
import vehicles  # our file
import time

from tkinter import *

count_lot = 0

root = Tk()
root.geometry("1600x800+0+0")
root.title(" Smart Vehicle Management System ")


def secs_diff(endTime, begTime):
    diff = (endTime - begTime).total_seconds()
    return diff


def get_speed(pixels, ftperpixel, secs):
    if secs > 0.0:
        return ((pixels * ftperpixel) / secs) * 0.681818
    else:
        return 0.0


def vid():
    car = cv2.CascadeClassifier("cars.xml")
    cap = cv2.VideoCapture("cars.mp4")
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # ty = cv2.BackgroundSubtractorMOG2()
    initial_time = datetime.datetime.now()
    c = 0
    b = []
    v = 0
    FOV = 53.5
    DISTANCE = 76
    IMAGEWIDTH = 640
    mph = 0
    frame_width_ft = 2 * (math.tan(math.radians(FOV * 0.5)) * DISTANCE)
    ftperpixel = frame_width_ft / float(IMAGEWIDTH)

    for i in range(0, 1000000, +1):
        new = []
        for j in range(0, 2):
            new.append(0)
        b.append(new)
    u = 0
    while True:

        ret, img = cap.read()
        initial_time = datetime.datetime.now()
        ##    print timestamp
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fe = car.detectMultiScale(gray, 1.1, 5)
        ##  cv2.line(img,(400,100),(1000,100),(200,200,0),2)
        ##  cv2.rectangle(img,(300,100),(1200,300),(0,255,0),3)
        c = 0
        v += 1
        u += 1
        for (x, y, w, h) in fe:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
            timestamp = datetime.datetime.now()
            secs = secs_diff(timestamp, initial_time)
            x1 = w / 2
            y1 = h / 2
            cx = x + int(x1)
            cy = y + int(y1)
            centroid = (cx, cy)
            ##        print cy
            if (int(cy) > b[c][1]):
                diff = cy - b[c][1]
            mph = get_speed(diff, ftperpixel, secs)
            mph *= 100
            if (mph > 1000):
                mph /= 100
            elif (mph > 100):
                mph /= 10
            mph = int(mph)
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (cx, cy)
            fontScale = 1
            fontColor = (255, 255, 0)
            lineType = 2

            cv2.putText(img, str(mph) + 'km/h', bottomLeftCornerOfText, font, fontScale, fontColor, lineType,
                        cv2.LINE_AA)
            cv2.imshow('Speed Detection', img)

            b[c][0] = int(centroid[0])
            b[c][1] = int(centroid[1])
            c += 1

        k = cv2.waitKey(100)
        if (u <= 299):
            print(u)
        else:
            break
        if (k == 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def mov():
    cap = cv2.VideoCapture("cars.mp4")
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        # cv2.imshow('diff',diff)

        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray',gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # cv2.imshow('blur',blur)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        # cv2.imshow('thresh',thresh)
        dilated = cv2.dilate(thresh, None, iterations=4)
        # cv2.imshow('dilated',dilated)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(fr ame1, contours, -1, (0,255,0), 3)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 1000:
                continue
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame1, "status:{}".format('Moving'), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 225), 4)

        cv2.imshow('Movement', frame1)
        frame1 = frame2
        ret, frame2 = cap.read()
        if ret == False:
            break
        if cv2.waitKey(20) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


def count_dir():
    cnt_up = 0
    cnt_down = 0
    cnt_right = 0
    cnt_left = 0

    cap = cv2.VideoCapture("cars.mp4")

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

            cv2.imshow('Counting and Detection', frame)

            if cv2.waitKey(10) == 27:
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def empty_slot_detection():
    # car_not_in_lot = cv2.imread("parking-no-car.png")

    # Selecting ROI
    global count_lot
    if count_lot % 2 == 0:
        car_in_lot = cv2.imread("parking-carr.png", 0)
    else:
        car_in_lot = cv2.imread("parking-car.png", 0)

    count_lot = count_lot + 1

    no = cv2.imread("no.png", 1)
    yes = cv2.imread("yes.png", 1)

    cv2.imshow(" Parking Lot ", car_in_lot)

    car = car_in_lot[200:400, 470:670]

    # Finding average color of the image
    avg_color_per_row = np.average(car, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    print(avg_color)

    # Detecting the car
    if avg_color < 140:
        print("Car is in the lot")
        cv2.imshow(" Sorry ", no)
    else:
        print("Car is not in the lot")
        cv2.imshow(" Welcome ", yes)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


topframe = Frame(root, width=1600, height=50)
topframe.pack()

lbl1 = Label(topframe, font=('arial', 50, 'bold'), bg="purple", text=" Smart Vehicle Management System ", fg="white",
             bd=10, anchor='w').pack()

# f3=Frame(root,width=400,height =20).pack()

f1 = Frame(root, width=400, height=20).pack()
localtime = str(datetime.date.today())
Label(f1, font=('arial', 20, 'bold'), text='Date: ' + localtime).pack()  # lebel which  is showed

f90_count = Frame(root, width=400, height=80)
f90_count.pack()
f4_mov = Frame(root, width=400, height=80)
f4_mov.pack()
button1 = Button(f4_mov, text="Check Movement", fg="blue", bg="black", font=('arial', 20, 'bold'), command=mov)
button1.pack()

f90_count = Frame(root, width=10, height=5)
f90_count.pack()
f1_speed = Frame(root, width=400, height=80)
f1_speed.pack()
button2 = Button(f1_speed, text="Check Speed", fg="yellow", bg="black", font=('arial', 20, 'bold'), command=vid)
button2.pack()

f90_count = Frame(root, width=10, height=5)
f90_count.pack()
f7_count = Frame(root, width=400, height=80)
f7_count.pack()
button3 = Button(f7_count, text="Check Counting With Direction ", fg="red", bg="black", font=('arial', 20, 'bold'),
                 command=count_dir)
button3.pack()

f90_count = Frame(root, width=10, height=5)
f90_count.pack()
f9_count = Frame(root, width=400, height=80)
f9_count.pack()
button4 = Button(f9_count, text="Check Parking Lot ", fg="green", bg="black", font=('arial', 20, 'bold'),
                 command=empty_slot_detection)
button4.pack()

f90_count = Frame(root, width=10, height=5)
f90_count.pack()
f6_quit = Frame(root, width=100, height=100)
f6_quit.pack()
button5 = Button(f6_quit, text="Q U I T", fg="white", bg="black", font=('arial', 20, 'bold'), command=quit)
button5.pack()

root.mainloop()

