import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time 

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 30
imgSize = 300  # Target image size
folder= "Data/C"
counter= 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image. Check your camera.")
        continue
    
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # White background

        # Ensure valid cropping region
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        # Get the shape of the cropped image
        hCrop, wCrop, _ = imgCrop.shape

        # Determine the scaling factor to fit within 300x300
        aspect_ratio = wCrop / hCrop

        if aspect_ratio > 1:  # Wider than tall
            new_w = imgSize
            new_h = int(imgSize / aspect_ratio)
        else:  # Taller than wide
            new_h = imgSize
            new_w = int(imgSize * aspect_ratio)

        imgResize = cv2.resize(imgCrop, (new_w, new_h))  # Resize while maintaining aspect ratio
        hResize, wResize, _ = imgResize.shape

        # Centering the resized image on the white canvas
        y_offset = (imgSize - hResize) // 2
        x_offset = (imgSize - wResize) // 2

        imgWhite[y_offset:y_offset + hResize, x_offset:x_offset + wResize] = imgResize

        cv2.imshow("imageCrop", imgCrop)
        cv2.imshow("imageWhite", imgWhite)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/C.{time.time()}.jpg', imgWhite)
        print(counter)
