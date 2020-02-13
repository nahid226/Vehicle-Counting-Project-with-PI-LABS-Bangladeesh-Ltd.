import tkinter
import cv2
import PIL.Image,PIL.ImageTk
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='path to the destination folder')
parser.add_argument('--source', help='source of the video stream')
args = parser.parse_args()


#empty_slot_var = "disabled"
#args.source = 0
#args.path = 'G:\\'
if args.source == '0':
    args.source = 0
#motion_enable = 0
class App:
    def __init__(self, window, window_title, video_source=args.source):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.grid(row=0, column=0, padx=20, pady=10)

        # Button that lets the user take a snapshot
        self.button_snapshot = tkinter.Button(window, text="Take Snapshot", width=50, command=self.snapshot)
        self.button_snapshot.grid(row=1, column=0, padx=10, pady=10)

        # Button that lets the user to detect movement
        self.button_movement_detect = tkinter.Button(window, text="Ditecting Movement", width=50, command=self.movement)
        self.button_movement_detect.grid(row=1, column=1, padx=10, pady=10)

        # Button that lets the user to check whether the slot is empty or not
        self.button_empty_slot = tkinter.Button(window, text="Empty Slot Checking", width=50, command=self.emptySlot)
        self.button_empty_slot.grid(row=1, column=2, padx=10, pady=10)

        #self.button_snapshot = tkinter.Button(window, text="Take Snapshot", width=50, command=self.snapshot)
        #self.button_snapshot.grid(row=2, column=0, padx=10, pady=10)

        #self.button_snapshot = tkinter.Button(window, text="Take Snapshot", width=50, command=self.snapshot)
        #self.button_snapshot.grid(row=2, column=1, padx=10, pady=10)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()


        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            cv2.imwrite(args.path+"frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)


    def movement(self):
        ret, frame1 = self.vid.get_frame()
        ret, frame2 = self.vid.get_frame()

        while self.vid.isOpened():
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
            # cv2.drawContours(frame1, contours, -1, (0,255,0), 3)
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                if cv2.contourArea(contour) < 1000:
                    continue
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame1, "status:{}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 225),
                            4)

            cv2.imshow('NAHID', frame1)
            frame1 = frame2
            ret, frame2 = cap.read()
            if ret == False:
                break
            if cv2.waitKey(1000) == 27:
                break

        cv2.destroyAllWindows()
        #self.vid.release()

    def emptySlot(self):
        img = cv2.imread("car.jpg")
        ret, thresh1 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)

        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite("carcount_closing.png", closing)
        cv2.imwrite("carcount_opening.png", opening)

class MyVideoCapture:

    def __init__(self, video_source=args.source):
        # open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open Video source", video_source)

        # get the video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    #release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()



#create a window and pass it to the application object
App(tkinter.Tk(), "Vehicle Counting and Direction")