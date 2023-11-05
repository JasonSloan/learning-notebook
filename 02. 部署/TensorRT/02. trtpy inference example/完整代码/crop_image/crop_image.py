import cv2
img = cv2.imread("OST_009.png",1)
print(img.shape)
img = cv2.resize(img, (428, 240))
print(img.shape)
cv2.imwrite("../workspace/OST_009_croped.jpg", img)