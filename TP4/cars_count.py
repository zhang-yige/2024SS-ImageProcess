import cv2
import numpy as np
from autoroute import moyenne_images

# 初始化视频读取器
cap = cv2.VideoCapture('TP4/video.avi')

# 创建用于检测特征点的FeatureDetector对象
feature_detector = cv2.SIFT_create()

# 读取视频的前两帧
ret, prev_frame = cap.read()
ret, next_frame = cap.read()

frames = []
gray_frames = []
while True:
    ret, im = cap.read()
    if not ret:
        break
    frames.append(im)
    imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray_frames.append(imGray)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# 初始化prevPts和trackedPts
M = 500
moyenne_img = moyenne_images(gray_frames, M)
_, remove_sky = cv2.threshold(moyenne_img, 210, 255, cv2.THRESH_TOZERO_INV)
# cv.imshow("Remove sky", remove_sky)
_, mask = cv2.threshold(remove_sky, 100, 255, cv2.THRESH_BINARY)
# cv.imshow("bilinairy image", mask)
kernel_open = np.ones((15,13), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
# cv.imshow("morphologyEx Open", mask)
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,25))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)


points = feature_detector.detect(moyenne_img)
prevPts = np.float32([p.pt for p in points if mask[int(p.pt[1]), int(p.pt[0])] == 255])

# 初始化trackedPts为None
trackedPts = None


while True:
    # 转换为灰度图像
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # 特征点检测
    keypoints_prev = feature_detector.detect(prev_gray)
    p0 = np.float32([keypoints_prev[i].pt for i in range(len(keypoints_prev))])

    # 计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None)

    # 选择状态为1的点，即跟踪成功的点
    good_old = p0[st==1]
    good_new = p1[st==1].reshape(-1, 2)

    # 绘制跟踪的点
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(next_frame, (a, b), (c, d), (0, 255, 0), 2)
        cv2.circle(next_frame, (a, b), 5, (0, 255, 0), -1)

    # 更新prev_frame和prevPts为下一帧的数据
    prev_frame = next_frame.copy()
    keypoints_next = feature_detector.detect(next_gray)
    prevPts = np.float32([keypoints_next[i].pt for i, kp in enumerate(keypoints_next) if kp in keypoints_prev])
    
    # 读取下一帧
    ret, next_frame = cap.read()

    # 如果没有下一帧，或者读取失败，退出循环
    if not ret:
        break

    # 显示光流结果
    cv2.imshow('Optical Flow', next_frame)

    # 按 'q' 退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频读取器和所有OpenCV窗口
cap.release()
cv2.destroyAllWindows()
