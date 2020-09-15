#!/usr/bin/python3
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

#hand = cv2.imread("~/mpv-shot0015.jpg",cv2.IMREAD_COLOR)
hand = cv2.imread("mugshot.jpg",cv2.IMREAD_COLOR)
hand_hsv = cv2.cvtColor(hand, cv2.COLOR_RGB2HSV)

h = hand_hsv[:, :, 0]
s = hand_hsv[:, :, 1]
v = hand_hsv[:, :, 2]



hand_flattened = []

for index in range(3):
    hand_flattened.append(np.array(hand_hsv[:, :, index]).flatten())

#print(hand_flattened.describe())

f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,5))
ax1.set_title("H")
ax1.hist(hand_flattened[0], bins=50)
ax2.set_title("S")
ax2.hist(hand_flattened[1], bins=50)
ax3.set_title("V")
ax3.hist(hand_flattened[2], bins=50)
plt.show()


low = []
high = []

def restrict(color_component):
    return np.clip(color_component, 0, 255)

z_value = 5.5

for i in range(3):
    mu = h[i].mean()
    sigma = h[i].std()
    deviation = z_value*sigma
    low.append(restrict(mu-deviation))
    high.append(restrict(mu+deviation))

print(low)
print(high)
