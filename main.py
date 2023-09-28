from tkinter import *
import cv2
import random
from PIL import ImageTk, Image
import numpy as np
import math
from cv2 import FONT_HERSHEY_SIMPLEX

#Defining Result Window
def startGame(yr_choice, dev_choice, result):
    your_choice = Label(win, text="Your Choice:", font=("Times", "20", "underline"), fg="blue", bg="lemon chiffon")
    your_choice.pack()
    your_choice.place(x=65, y=65)
    device_choice = Label(win, text="Device's Choice:", font=("Times", "20", "underline"), fg="blue", bg="lemon chiffon")
    device_choice.pack()
    device_choice.place(x=490, y=65)
    paper_path = "/Users/deepasharma/Downloads/Computer Vision/RockPaperScissor/paper.jpeg"
    rock_path = "/Users/deepasharma/Downloads/Computer Vision/RockPaperScissor/rock.jpeg"
    scissor_path = "/Users/deepasharma/Downloads/Computer Vision/RockPaperScissorrock.jpeg"
    if str(yr_choice) == "Paper":
        path = paper_path
        image1 = Image.open(path)
        test = ImageTk.PhotoImage(image1)
        label1 = Label(image=test)
        label1.image = test
        label1.pack()
        label1.place(x=50, y=100)
    if str(yr_choice) == "Rock":
        path = rock_path
        image1 = Image.open(path)
        test = ImageTk.PhotoImage(image1)
        label1 = Label(image=test)
        label1.image = test
        label1.pack()
        label1.place(x=50, y=100)
    if str(yr_choice) == "Scissor":
        path = scissor_path
        image1 = Image.open(path)
        test = ImageTk.PhotoImage(image1)
        label1 = Label(image=test)
        label1.image = test
        label1.pack()
        label1.place(x=50, y=100)
    if str(dev_choice) == "Paper":
        path = paper_path
        image1 = Image.open(path)
        test = ImageTk.PhotoImage(image1)
        label1 = Label(image=test)
        label1.image = test
        label1.pack()
        label1.place(x=485, y=100)
    if str(dev_choice) == "Rock":
        path = rock_path
        image1 = Image.open(path)
        test = ImageTk.PhotoImage(image1)
        label1 = Label(image=test)
        label1.image = test
        label1.pack()
        label1.place(x=485, y=100)
    if str(dev_choice) == "Scissor":
        path = scissor_path
        image1 = Image.open(path)
        test = ImageTk.PhotoImage(image1)
        label1 = Label(image=test)
        label1.image = test
        label1.pack()
        label1.place(x=485, y=100)
    canvas = Canvas(win,height=40,width=137,bg="#fff")
    canvas.pack()
    canvas.place(x=322, y=347)
    final_result = Label(win, text=result, font=("Times", "20"), fg="tomato", bg="lemon chiffon")
    final_result.pack()
    final_result.place(x=350, y=350)
#Create Game Window
def postureDetection():
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret,img = cap.read()
        final = []
        # Computer choices
        choices = ["Rock", "Paper", "Scissor"]
        comp_choice = random.choice(choices)

        # Create sub window in screen
        cv2.rectangle(img, (500, 500), (200, 200), (0, 255, 0), 0)
        crop_img = img[200:500, 200:500]

        # Convert image to gray scale
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # applying Gaussian blur
        value = (35, 35)
        blurred = cv2.GaussianBlur(gray, value, 0)

        # Thresholding
        _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Show thresholded image
        cv2.imshow('Thresholded', thresh1)

        # check OpenCV version to avoid unpacking error
        (version, _, _) = cv2.__version__.split('.')

        if version == '3':
            image, contours, hieraracy = cv2.findContours(thresh1.copy(), \
                                                          cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif version == '4':
            contours, hieraracy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, \
                                                   cv2.CHAIN_APPROX_NONE)

        # find contour with max area
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # Create bounding reactangle around the contour
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # approx the contour a little
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # finding convex hull
        hull = cv2.convexHull(cnt)

        # define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)

        # find the percentage of area not covered by convex hull
        arearatio = ((areahull - areacnt) / areacnt) * 100

        # drawing contours
        drawing = np.zeros(crop_img.shape, np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt, returnPoints=False)

        # finding convexity defects
        defects = cv2.convexityDefects(cnt, hull)
        # print(defects)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

        if defects is None:
            continue

        # applying Cosine Rule to find angle for all defects(between fingers)
        # with angle>90 degree and ignore defects
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

            # distance between point and convex hull
            d = (2 * ar) / a

            # apply cosine rule here
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            # ignore angle>90 and highlight rest red dots
            if angle <= 90 and d > 30:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0, 0, 225], -1)

            cv2.line(crop_img, start, end, [0, 255, 0], 2)

            # show appropriate image in window
        cv2.imshow('Gesture', img)
        all_img = np.hstack((drawing, crop_img))
        cv2.imshow('Contours', all_img)
        # cv2.waitKey(100)

        if areacnt < 20000 and count_defects == 0:
            user_choice = "Rock"
            if str(user_choice) == str(comp_choice):
                final.append("Game Draw")
            else:
                if str(comp_choice) == "Paper":
                    final.append("You Loose")
                if str(comp_choice) == "Scissor":
                    # print("You Win")
                    final.append("You Win")
            if user_choice in choices:
                startGame(user_choice, comp_choice, final[0])
            else:
                break
            
          

        elif count_defects<3 or arearatio>30:
            user_choice = "Scissor"
            if str(user_choice) == str(comp_choice):
                final.append("Game Draw")
            else:
                if str(comp_choice) == "Rock":
                    final.append("You Loose")
                if str(comp_choice) == "Paper":
                    final.append("You Win")

            if user_choice in choices:
                startGame(user_choice, comp_choice, final[0])
            else:
                break
        
                


        elif count_defects > 2 or arearatio >10:
            user_choice = "Paper"
            if str(user_choice) == str(comp_choice):
                final.append("Game Draw")
            else:
                if str(comp_choice) == "Rock":
                    final.append("You Win")
                if str(comp_choice) == "Scissor":
                    final.append("You Loose")
            if user_choice in choices:
                startGame(user_choice, comp_choice, final[0])
            else:
                break

    # k = cv2.waitKey(10)
    # if k == 27:
    #     breakpoint

    cv2.destroyAllWindows()
    cap.release()   

win = Tk()
win.geometry('720x500')
win.title("Rock Paper Scissor's Game")
win.configure(bg='light blue')
win.resizable(False, False)
win.iconbitmap('Brand-Logo-Icon.ico')
#App Name
appLabel = Label(win, text="Rock Paper Scissor", font=("Times", "35", "bold italic"), fg="dark blue", bg='lightblue', anchor=CENTER) #Label for Appname
appLabel.pack() #Packs appLabel
appLabel.place(x=50,y=5) #Position for appLabel
#Start Button
start_button = Button(win, text="Start Game", fg='green', bg='yellow', font=("Helvetica", "15", "italic"), bd='3',  command=postureDetection) #Creates Add Contact Button
start_button.place(x=310,y=175) #Position add_button
win.mainloop()