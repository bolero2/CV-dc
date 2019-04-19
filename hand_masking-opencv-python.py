import cv2 as cv

# img = cv.imread("C:\\Users\\Administrator\\Desktop\\testImage2.jpg")
cap = cv.VideoCapture(0)
cv.namedWindow("video")
cv.namedWindow("original")


def cvtimg(img):
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    range = cv.inRange(src=ycrcb, lowerb=(0, 133, 77), upperb=(255, 173, 127))
    blur = cv.GaussianBlur(src=range, ksize=(31, 31), sigmaX= 4)
    _, th1 = cv.threshold(src=blur, thresh=127, maxval=255, type=cv.THRESH_BINARY)

    return th1


ret, frame = cap.read()

while ret:
    ret, frame = cap.read()
    cv.imshow("original", frame)
    cv.imshow("video", cvtimg(frame))
    cv.waitKey(10)

exit(0)
