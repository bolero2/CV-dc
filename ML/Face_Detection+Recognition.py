import cv2
import numpy as np
import face_recognition
import dlib
import copy


def compare_faces_ordered(encodings, face_names, encoding_to_check):
    """Returns the ordered distances and names when comparing a list of face encodings against a candidate to check"""

    distances = list(np.linalg.norm(encodings - encoding_to_check, axis=1))
    return zip(*sorted(zip(distances, face_names)))


def compare_faces(encodings, encoding_to_check):
    """Returns the distances when comparing a list of face encodings against a candidate to check"""

    return list(np.linalg.norm(encodings - encoding_to_check, axis=1))


def face_encodings(face_image, number_of_times_to_upsample=1, num_jitters=1):
    """Returns the 128D descriptor for each face in the image"""

    # Detect faces:
    face_locations = detector(face_image, number_of_times_to_upsample)
    # Detected landmarks:
    raw_landmarks = [pose_predictor_5_point(face_image, face_location) for face_location in face_locations]
    # Calculate the face encoding for every detected face using the detected landmarks for each one:
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
            raw_landmark_set in raw_landmarks]


# 수정 부분
#######################################################################################################################
video = ["video5.mp4", "video1.mp4"]
wanted = ['moon2.jpg', "rv_irin.jpg", "rv_seulgi.jpg", "rv_yeri.jpg"]
#######################################################################################################################

# 1행 -> wanted 출현 횟수
# 2행 -> wanted 출현한 best matching distance
# 3행 -> wanted 출현한 best matching distance 의 시간(frame number)
is_watched = np.zeros(shape=(4, len(wanted)))

for video_number in video:
    out_video = "out_" + video_number[:-4] + ".avi"

    net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")

    # face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    pose_predictor_5_point = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    detector = dlib.get_frontal_face_detector()  # dlib hog face detector을 사용하였음.

    unknown_encoding_list = list()
    matching_distance_list = list()

    for wanted_num in wanted:
        target = cv2.imread(wanted_num)
        unknown_image = target[:, :, ::-1]
        unknown_encoding = face_encodings(unknown_image)[0]
        unknown_encoding_list.append(unknown_encoding)
        # print(unknown_encoding_list)

    # target_image = image[:, :, ::-1]
    # target_encoding = face_encodings(target_image)[0]

    cap = cv2.VideoCapture(video_number)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    running_time = total_frame / fps

    print("Total Video Frame={}".format(total_frame))
    print("FPS={}".format(round(fps, 2)))
    print("Video => [Width={0} | Height={1}]".format(width, height))
    print("Video Running time={}".format(str(round(running_time, 2)) + "[sec]"))
    ret, frame = cap.read()
    h, w, c = frame.shape

    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    out = cv2.VideoWriter(out_video, fourcc, int(fps), (int(width), int(height)))

    frame_count = 1

    while ret:
        # print(" * Now Frame count={}".format(frame_count))
        b = frame[:, :, 0]
        g = frame[:, :, 1]
        r = frame[:, :, 2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     [np.sum(b) / (b.shape[0] * b.shape[1]),
                                      np.sum(g) / (g.shape[0] * g.shape[1]),
                                      np.sum(r) / (r.shape[0] * r.shape[1])], False, False)
        # blob에서 mean 변수는 각 채널의 평균값을 빼줌.
        # np.sum(ch) / (ch.shape[0] * ch.shape[1])

        net.setInput(blob)
        detection = net.forward()
        detection = np.squeeze(detection)
        # print("detection->", detection)
        # print("detection shape : ", detection.shape)

        count = detection.shape[0]  # 검출된 box의 수(이 box는 얼굴일 수도 있고 아닐수도 있음)
        best_score = detection[0][2]
        # print("best score is {}".format(best_score))

        for i in range(0, count):  # 박스 별로 작업 진행
            matching_distance_list = list()
            if best_score < 0.2 and detection[i][2] >= best_score * 0.70:  # 낮은 확률인데 얼굴이 검출되는 경우
                # print("[{0} Face detected] confidence score : {1}".format(i, detection[i][2]))

                # 얼굴(Face)이라고 여겨지는 부분 추출
                face = frame[int(detection[i][4] * h):int(detection[i][6] * h),
                       int(detection[i][3] * w):int(detection[i][5] * w), :]

                face_landmarks_list_5 = face_recognition.face_landmarks(face, None, "small")  # 랜드마크 검출 시도
                if len(face_landmarks_list_5) != 0:  # 랜드마크가 존재하면 -> 거의 얼굴이라고 판단할 수 있음
                    face_landmarks_list_5 = face_landmarks_list_5[0]

                    left = face_landmarks_list_5['left_eye']
                    right = face_landmarks_list_5['right_eye']

                    known_image = face[:, :, ::-1]
                    known_encoding = face_encodings(face)
                    # print("known  encoding:", known_encoding)

                    if known_encoding:
                        known_encoding = known_encoding[0]
                        for wtd in range(0, len(unknown_encoding_list)):
                            computed_distances = compare_faces([known_encoding], unknown_encoding_list[wtd])
                            matching_distance_list.append(computed_distances[0])
                        most_similar = min(matching_distance_list)
                        # 박스 하나당 Similar한 사람은 단 한명 뿐. 거리가 가장 짧은 인덱스를 구한다.(그 거리는 0.4로 제한)
                        if most_similar <= 0.45:
                            finded = matching_distance_list.index(most_similar)

                            is_watched[0, finded] = is_watched[0, finded] + 1

                            before_similar = is_watched[1, finded]

                            if before_similar == float(0):  # 첫 등장일 경우 무조건 넣음
                                is_watched[1, finded] = most_similar
                                is_watched[2, finded] = frame_count
                                is_watched[3, finded] = frame_count
                            else:  # 한 번 등장한 이력이 있을경우
                                is_watched[3, finded] = is_watched[3, finded] + 1
                                if before_similar > most_similar:  # 이전 스코어보다 현재 스코어가 더 낮으면 추가
                                    is_watched[1, finded] = most_similar
                                    is_watched[2, finded] = frame_count
                                else:
                                    pass
                            # print(finded)
                            # print("Find->", wanted[finded])
                            # 빨간색 박스처리
                            cv2.rectangle(frame, (int(detection[i][3] * w), int(detection[i][4] * h)),
                                          (int(detection[i][5] * w), int(detection[i][6] * h)), (0, 0, 255), 3)
                            # confidence score 출력
                            cv2.putText(frame, str(round(detection[i][2], 2)),
                                        (int(detection[i][3] * w), int(detection[i][4] * h)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            # 이름 출력
                            cv2.putText(frame, wanted[finded][:-4],
                                        (int(detection[i][5] * w), int(detection[i][6] * h)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            break
                        # 거리가 가장 짧은 인덱스가 0.4 이상인 경우(wanted에 항목없음)
                        else:
                            # 눈 모자이크 처리
                            cv2.line(face, left[0], right[0], (0, 0, 0), 10)
                            # 파란색 박스처리
                            cv2.rectangle(frame, (int(detection[i][3] * w), int(detection[i][4] * h)),
                                          (int(detection[i][5] * w), int(detection[i][6] * h)), (255, 0, 0),
                                          3)
                            # confidence score 출력
                            cv2.putText(frame, str(round(detection[i][2], 2)),
                                        (int(detection[i][3] * w), int(detection[i][4] * h)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    pass

            elif detection[i][2] > 0.7:  # 높은 확률을 가진 얼굴(거의 실제)
                # print("[{0} Face detected] confidence score : {1}".format(i, detection[i][2]))

                # 얼굴(Face)이라고 여겨지는 부분 추출
                face = frame[int(detection[i][4] * h):int(detection[i][6] * h),
                       int(detection[i][3] * w):int(detection[i][5] * w), :]

                face_landmarks_list_5 = face_recognition.face_landmarks(face, None, "small")  # 랜드마크 검출 시도
                if len(face_landmarks_list_5) != 0:  # 랜드마크가 존재하면 -> 거의 얼굴이라고 판단할 수 있음
                    face_landmarks_list_5 = face_landmarks_list_5[0]

                    left = face_landmarks_list_5['left_eye']
                    right = face_landmarks_list_5['right_eye']

                    known_image = face[:, :, ::-1]
                    known_encoding = face_encodings(face)
                    # print("known  encoding:", known_encoding)

                    if known_encoding:
                        known_encoding = known_encoding[0]
                        for wtd in range(0, len(unknown_encoding_list)):
                            computed_distances = compare_faces([known_encoding], unknown_encoding_list[wtd])
                            matching_distance_list.append(computed_distances[0])
                        most_similar = min(matching_distance_list)
                        # 박스 하나당 Similar한 사람은 단 한명 뿐. 거리가 가장 짧은 인덱스를 구한다.(그 거리는 0.4로 제한)
                        if most_similar <= 0.45:
                            finded = matching_distance_list.index(most_similar)

                            is_watched[0, finded] = is_watched[0, finded] + 1

                            before_similar = is_watched[1, finded]

                            if before_similar == float(0):  # 첫 등장일 경우 무조건 넣음
                                is_watched[1, finded] = most_similar
                                is_watched[2, finded] = frame_count
                                is_watched[3, finded] = frame_count
                            else:  # 한 번 등장한 이력이 있을경우
                                is_watched[3, finded] = is_watched[3, finded] + 1
                                if before_similar > most_similar:  # 이전 스코어보다 현재 스코어가 더 낮으면 추가
                                    is_watched[1, finded] = most_similar
                                    is_watched[2, finded] = frame_count
                                else:
                                    pass
                            # print(finded)
                            # print("Find->", wanted[finded])
                            # 빨간색 박스처리
                            cv2.rectangle(frame, (int(detection[i][3] * w), int(detection[i][4] * h)),
                                          (int(detection[i][5] * w), int(detection[i][6] * h)), (0, 0, 255), 3)
                            # confidence score 출력
                            cv2.putText(frame, str(round(detection[i][2], 2)),
                                        (int(detection[i][3] * w), int(detection[i][4] * h)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            # 이름 출력
                            cv2.putText(frame, wanted[finded][:-4],
                                        (int(detection[i][5] * w), int(detection[i][6] * h)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            break
                        # 거리가 가장 짧은 인덱스가 0.4 이상인 경우(wanted에 항목없음)
                        else:
                            # 눈 모자이크 처리
                            cv2.line(face, left[0], right[0], (0, 0, 0), 10)
                            # 파란색 박스처리
                            cv2.rectangle(frame, (int(detection[i][3] * w), int(detection[i][4] * h)),
                                          (int(detection[i][5] * w), int(detection[i][6] * h)), (255, 0, 0),
                                          3)
                            # confidence score 출력
                            cv2.putText(frame, str(round(detection[i][2], 2)),
                                        (int(detection[i][3] * w), int(detection[i][4] * h)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        print("Encoding Failure!")
                        pass

                else:
                    pass
            else:  # 높은 확률로 얼굴이 아닌경우
                # print("There is nothing landmark!\n")
                pass

        out.write(frame)
        # # frame = cv2.resize(frame, (960, 540))
        cv2.imshow("video", frame)
        # cv2.imshow("face", face)
        cv2.waitKey(1)

        ret, frame = cap.read()
        frame_count += 1

    for ww in range(0, len(wanted)):
        if int(is_watched[0, ww]) != 0:
            print(
                f'{wanted[ww]} found: file={video_number}, total frame={is_watched[0, ww]}, best: score={is_watched[1, ww]:0.2f},'
                f' best match frame={int(is_watched[2, ww])}({int(is_watched[2, ww] / (fps * 60))}:{int(is_watched[2, ww] / fps):02d})')
    is_watched = list()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
