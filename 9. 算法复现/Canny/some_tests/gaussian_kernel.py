import numpy as np


def gaussian_kernel(size, sigma=1):
    k = (size - 1) / 2
    kernel = np.zeros([size, size])
    normal = 1 / (2 * np.pi * sigma ** 2)
    for i in range(1, size + 1):
        for j in range(1, size + 1):
            kernel[i - 1, j - 1] = normal * np.exp(-1 / (2 * sigma ** 2) * ((i - (k + 1)) ** 2 + (j - (k + 1)) ** 2))
    return kernel


if __name__ == '__main__':
    kernel = gaussian_kernel(3, 1)
    import cv2
    img = cv2.imread("./pics/cat.png", 0)
    result = cv2.filter2D(img, -1, kernel)

    # 显示原始图像和卷积结果
    cv2.imshow('Original Image', img)
    cv2.imshow('Convolution Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

