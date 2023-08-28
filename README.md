# Car and Its Number Plate Detection, and Pedestrian Detection using Digital Image Processing with Python



Welcome to our project repository focused on **Car and Its Number Plate Detection**, and **Pedestrian Detection** using **Digital Image Processing** with Python. This project employs various computer vision and image processing techniques to achieve object detection and recognition within images and videos.

## Table of Contents

- [Introduction](#introduction)
- [Motivation](#motivation)
- [Methodology](#methodology)
- [Algorithm Overview](#algorithm-overview)
- [Car Detection](#car-detection)
- [Number Plate Detection](#number-plate-detection)
- [Pedestrian Detection](#pedestrian-detection)
- [Requirements](#requirements)
- [Results](#results)
- [Contributors](#contributors)


## Introduction

This project focuses on leveraging digital image processing techniques for detecting and recognizing specific objects within images and videos. The main objectives are car and its number plate detection, as well as pedestrian detection.

## Motivation

The increasing need for automated systems and reduced human interactions, especially due to factors like the COVID-19 pandemic, has led to the development of technologies that minimize manual efforts and enhance safety. This project aims to contribute to this objective by implementing object detection and recognition capabilities in the context of cars, their number plates, and pedestrians.

## Methodology

The project follows a systematic approach to achieve the targeted object detection and recognition:

1. **Car Detection using Haar Cascade:** The Haar Cascade technique is applied to detect cars within images. This technique employs a cascade of classifiers trained to identify specific car features. Detected cars are marked with bounding rectangles.

2. **Number Plate Detection using Canny Filter and Contours:** The Canny edge detection algorithm is used to locate edges in the image. By identifying contours, potential regions containing number plates are pinpointed. When contours are detected, rectangles are drawn around the number plates.

3. **Pedestrian Detection using Histogram of Oriented Gradients (HOG):** For pedestrian detection, the Histogram of Oriented Gradients (HOG) descriptor is utilized. This descriptor captures features related to human shape and structure. The SVM classifier aids in identifying pedestrians based on these features.

## Flow Diagram

<div align="center">
  <img src="https://i.imgur.com/ZBh01BX.jpg" alt="Flow Diagram">
</div>

## Algorithm Overview

The project algorithm involves multiple stages, including car detection using Haar Cascade, number plate detection using edge detection and contours, and pedestrian detection using the HOG descriptor. 

## Car Detection

In this section, we use the Haar Cascade technique to detect cars within images. The Haar Cascade method employs classifiers trained on specific features of cars. When a car is detected, a bounding rectangle is drawn around it, and it's labeled as "Car."

```python
# Car Detection using Haar Cascade
cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(10, 10))
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(img, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2) 
## Number Plate Detection
```
In this section, we apply the Canny edge detection algorithm and contour analysis to detect number plates within the images. The algorithm identifies potential regions containing number plates and draws rectangles around them.

```python
# Number Plate Detection using Canny Filter and Contours
gray = cv2.bilateralFilter(gray, 13, 15, 15)
edged = cv2.Canny(gray, 30, 200)
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print("No contour detected")
else:
    detected = 1
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
```
## Number Plate Detection

In this section, we apply the Canny edge detection algorithm and contour analysis to detect number plates within the images. The algorithm identifies potential regions containing number plates and draws rectangles around them.

```python
# Number Plate Detection using Canny Filter and Contours
gray = cv2.bilateralFilter(gray, 13, 15, 15)
edged = cv2.Canny(gray, 30, 200)
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print("No contour detected")
else:
    detected = 1
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
```
## Pedestrian Detection

In this section, we use the Histogram of Oriented Gradients (HOG) descriptor along with SVM classification to detect pedestrians within the images.

```python
# Pedestrian Detection using Histogram of Oriented Gradients (HOG)
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

rects, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)
for x, y, w, h in zip(rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (139, 34, 104), 2)
## Requirements

To run this project, you'll need:

- Python 3.x
- OpenCV library
- NumPy library
- Imutils library
```

## Results

Our project demonstrates effective object detection and recognition capabilities. The algorithms continuously improve accuracy and reliability. Check out the sample input and output showcasing our object detection capabilities:

<div align="center">
  <h3>Sample Input</h3>
  <img src="https://i.imgur.com/rCLRm9J.jpg" alt="Sample Input">
</div>

<div align="center">
  <h4>Output 1: Detecting Car and Pedestrian</h4>
  <img src="https://i.imgur.com/av1oMsS.jpg" alt="Car and Pedestrian Detection">
</div>

Detection of both a car and a pedestrian in the input image.

<div align="center">
  <h4>Output 2: Detecting Number Plate</h4>
  <img src="https://i.imgur.com/5th86t8.jpg" alt="Number Plate Detection">
</div>

Detection of a number plate in the input image.

## Contributors



- [Mohid Aamir](https://github.com/arigatohaxroot) 
- [Haseeb Mehmood]

Their dedication and expertise have been instrumental in making this project a success.



