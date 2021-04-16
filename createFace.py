import cv2
import numpy as numpy
import os, time

FACE_DIR = "data/train/"


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def main():
    create_folder(FACE_DIR)
    while True:
        name=input("EnterName: ")
        try:
            face_folder = FACE_DIR + name + "/"
            create_folder(face_folder)
            break
        except:
            print("Invalid input.!")
            continue

    while True:
        init_img_no = input("Starting img no.: ")
        try:
            init_img_no = int(init_img_no)
            break
        except:
            print("Starting img no should be integer...")
            continue

    img_no = init_img_no
    cap = cv2.VideoCapture(0)
    total_imgs = 20
    while True:
        ret, img = cap.read()
        img_path = face_folder +name+ str(img_no) + ".jpg"
        cv2.imwrite(img_path, img)
        cv2.imshow("aligned", img)
        img_no += 1

        cv2.imshow("Saving", img)
        cv2.waitKey(100)
        if img_no == init_img_no + total_imgs:
            break
    cap.release()
main()