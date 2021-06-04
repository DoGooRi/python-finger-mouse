import cv2
import numpy as np
import hand_tracking as htm
import time
import autopy

# 웹캠 설정
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 2

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# 0번 웹캠 캡쳐
cap = cv2.VideoCapture(0)
# 640 x 480 set
cap.set(3, wCam)
cap.set(4, hCam)
# 핸드 트래킹 호출
detector = htm.handDetector(maxHands=1)
# 모니터 화면 사이즈
wScr, hScr = autopy.screen.size()

while True:
    # 손 추적 및 그리기
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 감지된 손가락 끝 포인트가 0개가 아니라면 (감지됐다면)
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:] # 검지 끝 좌표
        x2, y2 = lmList[12][1:] # 중지 끝 좌표
        # print(x1, y1, x2, y2)

        # 핀 손가락 체크
        fingers = detector.fingersUp()
        print(fingers)

        # 마우스 움직임 구역 그리기
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 검지만 핀 상태
        if fingers[1] == 1 and fingers[2] == 0:
            # 좌표 값 선형 보간 처리 (움직임 부드럽게 하기 위해)
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 부드럽게 처리
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 마우스 움직임 처리
            autopy.mouse.move(wScr - clocX, clocY)

            # 마우스 포인터 그리기 (검지 끝 좌표)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 검지와 중지 둘다 핀 상태
        if fingers[1] == 1 and fingers[2] == 1:
            # 검지와 중지 사이의 거리 구하기
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            # 거리가 30 미만이라면 클릭
            if length < 30:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

        # 검지 ~ 약지 모두 필 경우 프로그램 종료
        if fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
            break

    # 프레임 레이트 표기
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 캠 화면 출력
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
        break
