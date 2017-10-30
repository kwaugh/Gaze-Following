import cv2
import numpy as np

# Solve for an affine transformation (with scale, so 7 params)
# given a source and target frame
# returns rotation, translation, sigmoid

def affine(source, target):
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.SIFT()
    source_kp = sift.detect(source_gray, None) # None is mask
    target_kp = sift.detect(target_gray, None)

    homography, mask = cv2.findHomography(source_kp, target_kp, cv2.CV_RANSAC)

    print('homography:', homography)
    print('mask:', mask)

    return None, None, None

def main():
    affine(cv2.imread('left.jpg'), cv2.imread('right.jpg'))

if __name__ == '__main__':
    main()

