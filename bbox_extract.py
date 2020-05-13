import multiprocessing as mp
import cv2
import numpy as np
import os


def draw_bbox(file_name):

    img = cv2.pyrDown(cv2.imread(file_name))
    # print(img.shape)
    # img =cv2.resize(img,None,fx=0.2,fy=0.2)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray=img[:,:,1:2]
    # print(img.shape)
    gray=cv2.GaussianBlur(gray, (5, 5), 50)
    ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow("Image", 0)
    # cv2.imshow("Image", thresh)
    # 腐蚀图像
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=5)
    eroded = cv2.erode(dilated, kernel,iterations=5)
    # 膨胀图像

    # cv2.namedWindow("Dilated Image", 0)
    # cv2.imshow("Dilated Image", dilated)
    # cv2.namedWindow("eroded Image", 0)
    # cv2.imshow("eroded Image", eroded)

    img_backgorund=np.zeros_like(img)

    thresh = cv2.Canny(eroded, 0, 50)
    thresh = cv2.dilate(thresh, kernel, iterations=5)
    # cv2.namedWindow("thresh", 0)
    # cv2.imshow("thresh", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(thresh, contours, -1, (0, 255, 255), 2)

    # 找到最大区域并填充
    area = []
    for i in range(len(contours)):
        epsilon = 0.01 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)
        cv2.polylines(thresh, [approx, ], True, (0, 255, 0), 2)  # green
        area.append(cv2.contourArea(contours[i]))
        # print(area)
    # cv2.namedWindow("thresh", 0)
    # cv2.imshow("thresh", thresh)
    max_idx = np.argmax(area)
    # print(max_idx)
    cnt = contours[max_idx]
    x, y, w, h = cv2.boundingRect(cnt)
    return (x,y,w,h),img
    # print(x,y,w,h)
    # cv2.rectangle(img,(x-20,y-20),(x+w+20,y+h+20),color=(0,0,255),lineType=cv2.LINE_AA)
    # cv2.putText(img,label,(x-20,y-20),fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255,0,0))
    # # break
    # # cv2.namedWindow("rectangle", 0)
    # # cv2.imshow("rectangle", img)
    # folder_path, file_name = os.path.split(file_name)
    # cv2.imwrite(os.path.join('output',file_name),img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return (x,y,w,h),img


if __name__ == '__main__':
    draw_bbox('test/IMG_20200509_185605_1.jpg','good')


