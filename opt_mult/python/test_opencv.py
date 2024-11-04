import cv2
import numpy as np

random_image = np.zeros((100, 100))

cv2.imshow("title", random_image)
cv2.waitKey(0)
cv2.destroyAllWindows()