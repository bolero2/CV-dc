import cv2 as cv
import numpy as np

# img = cv.imread("C:\\Users\\Administrator\\Desktop\\testImage2.jpg")
cap = cv.VideoCapture(0)
cv.namedWindow("Threshold")
cv.namedWindow("YCrCb")
cv.namedWindow("original")


def cvtimg(img):
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    range = cv.inRange(src=ycrcb, lowerb=(0, 133, 77), upperb=(255, 173, 127))
    blur = cv.GaussianBlur(src=range, ksize=(31, 31), sigmaX=5)
    _, th1 = cv.threshold(src=blur, thresh=127, maxval=255, type=cv.THRESH_BINARY)

    return ycrcb, th1


ret, frame = cap.read()

while ret:
    ret, frame = cap.read()
    ycrcb, th = cvtimg(frame)
    cv.imshow("Threshold", th)
    cv.imshow("YCrCb", ycrcb)
    cv.imshow("original", frame)
    cv.waitKey(10)

exit(0)
