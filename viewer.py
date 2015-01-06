import sys
import cv2

img = cv2.imread(sys.argv[1])

cv2.namedWindow('prvi', cv2.WINDOW_AUTOSIZE)
cv2.imshow('prvi', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
