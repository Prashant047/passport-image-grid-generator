import cv2
import numpy as np
import sys

FACE_WIDTH_TO_PHOTO_WIDTH = 0.47
CENTER_SHIFT_FRACTION = 5
FINAL_PHOTO_WIDTH = 192
PADDING = 10

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

image = cv2.imread(sys.argv[1])
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(image_gray, 1.1, 4)

selected_image_box = [0,0,0,0]
for (x, y, w, h) in faces:
  if w*h > selected_image_box[2]*selected_image_box[3]:
    selected_image_box[2] = w
    selected_image_box[3] = h

    selected_image_box[0] = x
    selected_image_box[1] = y

[x, y, w, h] = selected_image_box

photo_box = [int(x-w/2), int(y-w/2 + w/CENTER_SHIFT_FRACTION), int(w/FACE_WIDTH_TO_PHOTO_WIDTH), int(w/FACE_WIDTH_TO_PHOTO_WIDTH)]
[xp, yp, wp, hp] = photo_box

cropped_image = image[yp:yp+hp, xp:xp+wp]
resized_image = cv2.resize(cropped_image, (FINAL_PHOTO_WIDTH,FINAL_PHOTO_WIDTH), interpolation=cv2.INTER_AREA)

a4_height, a4_width = int(11.7*96), int(8.3*96)
a4_image = np.zeros((a4_height, a4_width, 3), dtype="uint8")
a4_image[:,:] = [255,255,255]

for row in range(5):
  for col in range(3):
    a4_image[
      row*FINAL_PHOTO_WIDTH+PADDING*(1+row):row*FINAL_PHOTO_WIDTH+PADDING*(1+row)+FINAL_PHOTO_WIDTH,
      col*FINAL_PHOTO_WIDTH+PADDING*(1+col):col*FINAL_PHOTO_WIDTH+PADDING*(1+col)+FINAL_PHOTO_WIDTH,
      :
    ] = resized_image
    
cv2.imwrite('print.jpg', a4_image)

# cv2.imshow('image', a4_image)
# cv2.waitKey()