import cv2

cap = cv2.VideoCapture(0)

tracker = cv2.TrackerCSRT_create()
ret, img = cap.read()
bbox = cv2.selectROI("test", img)
print(f'selected ROI= {bbox}')
tracker.init(img, bbox)


def drawBox(img, bbox):
    print(bbox)
    return cv2.rectangle(img, pt1=(int(bbox[0]), int(bbox[1])), pt2=(int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])), color=(0, 0, 255), thickness=3)


while ret:
    timer = cv2.getTickCount()
    ret, img = cap.read()

    ret, bbox = tracker.update(img)
    
    if ret:
        img = drawBox(img, bbox)
    else:
        cv2.putText(img, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, f"FPS : {str(int(fps))}", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("test", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
