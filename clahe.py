import cv2
from PIL import Image

clipLimit = 1
tileGridsize = (11,11)
image = cv2.imread('2.jpg')

clahe = cv2.createCLAHE(clipLimit, tileGridsize)

ycrcb_array = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
y, cr, cb = cv2.split(ycrcb_array)
merge_array = cv2.merge([clahe.apply(y), cr, cb])
output = cv2.cvtColor(merge_array, cv2.COLOR_YCrCb2RGB)

cv2.imshow(".",output)
cv2.waitKey(0)