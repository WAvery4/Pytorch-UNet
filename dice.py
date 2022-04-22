import cv2
import os
import numpy as np

validation_size = 1

def dice(p, m, k = 1):
    intersect = np.sum(p[true==k]) * 2.0
    dice = intersect / (np.sum(p) + np.sum(true))
    return dice

scores = np.zeros(validation_size)
path = "./Bentonite-Data/Test/Mask/"
for i, image_path in enumerate(os.listdir(path)):
    pred = cv2.imread('./Bentonite-Data/Test/Pred/' + image_path)
    mask = cv2.imread('./Bentonite-Data/Test/Mask/' + image_path)

    scores[i] = dice(pred, mask)

print(np.average(scores))