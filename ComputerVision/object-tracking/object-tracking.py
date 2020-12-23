import cv2

cap = cv2.VideoCapture(0)

tracker = cv2.TrackerMOSSE_create()
ret, img = cap.read()
bbox = cv2.selectROI("test", img, False)
tracker.init(img, bbox)


def drawBox():
    pass


while ret:
    timer = cv2.getTickCount()
    ret, img = cap.read()

    ret, bbox = tracker.update(img)
    if ret:
        drawBox()
    else:
        cv2.putText(img, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, f"FPS : {str(int(fps))}", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("test", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
