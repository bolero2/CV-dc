import cv2
import threading as th
from pynput.keyboard import Listener, Key
from pynput import keyboard
import os
import glob


key = ''
yolo_coord = list()


def handlePress(_key):
    print(f"Press: {_key}")
    key = _key


def handleRelease(_key):
    print(f"Release: {_key}")
    key = _key


def draw_bbox(img, bbox):
    return cv2.rectangle(img, pt1=(int(bbox[0]), int(bbox[1])), pt2=(int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])), color=(0, 0, 255), thickness=3)


def xywh2yolo(img, bbox):
    class_num = 0
    xmin, ymin, width, height = bbox
    img_col, img_row, img_ch = img.shape[0:3]

    center_x = round((xmin + (width / 2)) / img_col, 4)
    center_y = round((ymin + (height / 2)) / img_row, 4)
    width = round(width / img_col, 4)
    height = round(height / img_row, 4)

    sentence = f'{class_num} {center_x} {center_y} {width} {height}\n' 

    return sentence


path = '/home/bolero/dcjs/image/'
filelist = sorted(os.listdir(path))
filecount = len(filelist)
count = 0

with Listener(on_press=handlePress, on_release=handleRelease) as listener:
    thread1 = th.Thread(target=listener.join, args=())
    thread1.daemon=True
    thread1.start()

    while count < filecount:
        img = cv2.imread(f'{path}{filelist[count]}')
        yolo_coord = list()
        if img is not None:
            f = open(f'{path}{filelist[count][:-3]}txt', 'w')
            while True:
                bbox = cv2.selectROI("image", img)
                print(f"ROI Bndbox Point= {bbox}")

                if len(bbox) != 0:
                    img = draw_bbox(img, bbox)
                else:
                    print("Select bndbox!")
                    continue
                   
                sentence = xywh2yolo(img, bbox)
                if sum(sentence) != 0:
                    yolo_coord.append(sentence)
                    print(yolo_coord)

                cv2.imshow("image", img)
                key = cv2.waitKey(0)
                if key == ord('s'):    # input (s)ave
                    print("Coordinate will be saved.")
                    break

            f.writelines(yolo_coord)
            print("Coordinate was saved")
            f.close()
            cv2.imshow("image", img)
            key = cv2.waitKey(0)
            if key == ord('a'):
                count = count - 1 
            elif key == ord('q'):
                print("Program is terminated")
                exit(0)
            if key == ord('d'):
                count = count + 1
                continue
