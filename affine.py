import cv2
import numpy as np
import config
import torch

# this function taken taken from
# https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
def detectAndDescribe(image):
    # image is of shape (3, 227, 227)
    image = np.swapaxes(image, 0, 1)
    image = np.swapaxes(image, 1, 2)
    # image is now of shape (227, 227, 3)
    # print('about to convert to grayscale')
    # print('image.shape:', image.shape)
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print('converted to grayscale')
    # print('gray.shape:', gray.shape)
    # print('gray before int conversion:', gray)
    # convert values for [0, 1] (float) to [0, 255] (int)

    gray = 255 * gray
    gray = cv2.convertScaleAbs(gray)
    # print('gray after int conversion:', gray)

    # check to see if we are using OpenCV 3.X
    if cv2.__version__ != '2.4.13.4':
        # detect and extract features from the image
        descriptor = cv2.xfeatures2d.SIFT_create()
        # print('created sift descriptor object')
        step_size = 5
        mask = np.asarray([cv2.KeyPoint(x, y, step_size)
                for y in range(0, gray.shape[0], step_size)
                for x in range(0, gray.shape[1], step_size)])
        # dense features
        (kps, features) = descriptor.compute(gray, mask)

        # sparse features
        # (kps, features) = descriptor.detectAndCompute(gray, None)
        # print('computed sift features')

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
    # print('getting affine transformation')
    numpy_source = source.data.cpu().numpy()
    numpy_target = target.data.cpu().numpy()

    translation = np.zeros((numpy_source.shape[0], 3))
    rotation = np.zeros((numpy_source.shape[0], 3, 3))
    sigmoid = np.zeros(numpy_source.shape[0])

    # TODO: consider doing this on the GPU because CPU side it'll be hella slow
    for i in range(numpy_source.shape[0]):
        (kps1, feats1) = detectAndDescribe(numpy_source[i])
        # print('found feats1')
        (kps2, feats2) = detectAndDescribe(numpy_target[i])
        # print('found feats2')

        if feats1 is None or feats2 is None:
            # print('COULD NOT FIND SIFT FEATURES')
            sigmoid[i] = 0
            continue
        # print()

        # print('numpy_target[i]:', numpy_target[i])

        # print('feats1.shape:', feats1.shape)
        # print('feats2.shape:', feats2.shape)
        M = matchKeypoints(kps1, kps2, feats1, feats2, ratio=0.75, reprojThresh=4.0)

        if M is None or M[1] is None:
            # print('COULD NOT SOLVE FOR HOMOGRAPHY')
            sigmoid[i] = 0
            continue

        # print('found a homography')

        (matches, H, status) = M
        scale = H[2][2]
        t = H[0:2, 2]
        r = H[0:2, 0:2] / scale

        # TODO: fix this because translation is 3 vector and t is 2 vector
        translation[i, 0:2] = t
        translation[i, 2] = 0

        # TODO: fix this because rotation is 3x3 and t is 2x2
        rotation[i, 0:2, 0:2] = r
        rotation[i, 0, 2] = 0
        rotation[i, 1, 2] = 0
        rotation[i, 2, 0] = 0
        rotation[i, 2, 1] = 0
        rotation[i, 2, 2] = 0


        sigmoid[i] = 1

    rotation_cuda = torch.from_numpy(rotation).type(torch.FloatTensor).cuda()
    translation_cuda = torch.from_numpy(translation).type(torch.FloatTensor).cuda()
    sigmoid_cuda = torch.from_numpy(sigmoid).type(torch.FloatTensor).cuda()

    # print('rotation_cuda.type():', rotation_cuda.type())
    # print('translation_cuda.type():', translation_cuda.type())
    # print('sigmoid_cuda.type():', sigmoid_cuda.type())
    # print('rotation_cuda:', rotation_cuda)

    return torch.autograd.Variable(rotation_cuda), torch.autograd.Variable(translation_cuda), torch.autograd.Variable(sigmoid_cuda)

def main():
    affine(cv2.resize(cv2.imread('left.jpg'), (224, 224)),
           cv2.resize(cv2.imread('right.jpg'), (224, 224)))

if __name__ == '__main__':
    main()

