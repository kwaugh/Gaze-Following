import cv2
import numpy as np

# this function taken taken from
# https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
def detectAndDescribe(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check to see if we are using OpenCV 3.X
    if cv2.__version__ != '2.4.13.4':
        # detect and extract features from the image
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)

    # otherwise, we are using OpenCV 2.4.X
    else:
        # detect keypoints in the image
        detector = cv2.FeatureDetector_create("SIFT")
        kps = detector.detect(gray)

        # extract features from the image
        extractor = cv2.DescriptorExtractor_create("SIFT")
        (kps, features) = extractor.compute(gray, kps)

    # convert the keypoints from KeyPoint objects to NumPy
    # arrays
    kps = np.float32([kp.pt for kp in kps])

    # return a tuple of keypoints and features
    return (kps, features)

# this function taken taken from
# https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    # compute the raw matches and initialize the list of actual
    # matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    # loop over the raw matches
    for m in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
            # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                    reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

    # otherwise, no homograpy could be computed
    return None

# Solve for an affine transformation (with scale, so 7 params)
# given a source and target frame
# returns rotation, translation, sigmoid
def affine(source, target):
    (kps1, feats1) = detectAndDescribe(source)
    (kps2, feats2) = detectAndDescribe(target)

    M = matchKeypoints(kps1, kps2, feats1, feats2, ratio=0.75, reprojThresh=4.0)
    if M is None:
        print('COULD NOT SOLVE FOR HOMOGRAPHY')
        return None, None, None

    (matches, H, status) = M

    scale = H[2][2]
    translation = H[0:2, 2]
    rotation = H[0:2, 0:2] / scale
    # TODO: I don't think this is valid. looks like their other param is
    # something else

    # replicate torch.nn.Hardtanh
    sigmoid = None
    if scale > 1:
        sigmoid = 1
    elif scale < 0:
        sigmoid = 0
    else:
        sigmoid = scale

    return rotation, translation, sigmoid

def main():
    affine(cv2.resize(cv2.imread('left.jpg'), (224, 224)),
           cv2.resize(cv2.imread('right.jpg'), (224, 224)))

if __name__ == '__main__':
    main()

