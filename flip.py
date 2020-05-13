import multiprocessing as mp
import cv2
import numpy as np
import os


def flip(file_name):

    img = cv2.imread(file_name)
    img=img[0:3000,:,:]
    cv2.namedWindow("flip", 0)
    cv2.imshow("flip", img)
    cv2.imwrite(file_name,img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    for idx, img in enumerate(os.listdir('test')):
        img = os.path.join('test', img)
        flip(img)