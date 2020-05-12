import multiprocessing as mp
import cv2
import numpy as np
import os


def run():

    img = cv2.pyrDown(cv2.imread('data/1/IMG_20200509_150624.jpg'))
    print(img.shape)
    # img =cv2.resize(img,None,fx=0.2,fy=0.2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (5, 5), 50)
    ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow("Image", 0)
    # cv2.imshow("Image", thresh)
    # 腐蚀图像
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(thresh, kernel,iterations=10)
    # 膨胀图像
    dilated = cv2.dilate(eroded, kernel,iterations=15)
    # cv2.namedWindow("Dilated Image", 0)
    # cv2.imshow("Dilated Image", dilated)
    img_backgorund=np.zeros_like(img)

    thresh = cv2.Canny(dilated, 0, 50)
    cv2.namedWindow("thresh", 0)
    cv2.imshow("thresh", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thresh, contours, -1, (0, 255, 255), 2)
    # 找到最大区域并填充
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)

    cnt = contours[max_idx]
    x, y, w, h = cv2.boundingRect(cnt)
    print(x,y,w,h)
    cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),lineType=cv2.LINE_AA)
        # cv2.drawContours(image, [x,y,w,h], -1, (0, 255, 0), 3)

    # approx = cv2.approxPolyDP(cnt, epsilon, True)
    # hull = cv2.convexHull(cnt)
    #
    # x, y, w, h = cv2.boundingRect(cnt)
    # min_rect = cv2.minAreaRect(cnt)
    # (x, y), radius = cv2.minEnclosingCircle(cnt)
    cv2.namedWindow('drawing',0)
    cv2.imshow('drawing', img)
    cv2.imwrite('test.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()



