import cv2
import numpy as np


class ImageFrame:
    def __init__(self, path, frame, K, dist, orb):
        self.path = path
        self.frame = frame
        self.K = K
        self.dist = dist
        self.orb = orb

    def imread(self):
        self.image = cv2.imread(self.path)

    def imshow(self):
        displayed_image = self.image
        for point in self.matched_points:
            point = np.int_(point)
            cv2.circle(displayed_image, (point[0], point[1]), 2, (0, 255, 0), -1)
        cv2.imshow(f"frame", displayed_image)
        return

    def undistort(self):
        self.image = cv2.undistort(self.image, self.K, self.dist)
        return

    def detect_features(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.key_points, self.descriptors = self.orb.detectAndCompute(gray_image, None)
        return

    def match_key_points(self, matches, main=True):
        self.matched_points = np.array(
            [self.key_points[m.trainIdx].pt for m in matches] if main else \
            [self.key_points[m.queryIdx].pt for m in matches]
        )
        self.matched_points = self.matched_points[:int(len(self.matched_points) * 0.85)]
        return
