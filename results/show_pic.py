import cv2
import sys

path = sys.argv[1]
print(path)
try:
    pic = cv2.imread(path)
    cv2.imshow("result", pic)
except Exception as e:
    print(e)
