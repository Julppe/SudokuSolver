import sys
import numpy as np
import cv2

im = cv2.imread('CombinedTraining1.PNG')

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

samples = np.empty((0, 400))
responses = []
keys = [i for i in range(48, 58)]

for area in contours:
    if cv2.contourArea(area) > 50:
        [x, y, width, height] = cv2.boundingRect(area)

        if 200 > height > 20 and width > 10:
            cv2.rectangle(im, (x, y), (x + width, y + height), (0, 0, 255), 2)
            roi = thresh[y:y + height, x:x + width]
            roi = cv2.resize(roi, (20, 20))
            cv2.imshow('norm', im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roi.reshape((1, 400))
                samples = np.append(samples, sample, 0)

responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
print("training complete")

np.savetxt('samples.data', samples)
np.savetxt('responses.data', responses)
