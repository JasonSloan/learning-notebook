from scipy import ndimage
import numpy as np

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta


if __name__ == '__main__':
    import cv2

    img = cv2.imread("./pics/person.png", 0)
    from gaussian_kernel import gaussian_kernel

    kernel = gaussian_kernel(3, 1.4)
    img = cv2.filter2D(img, -1, kernel)
    G, theta = sobel_filters(img)
    # G, theta = gradient_calculate(img)
    print(G[100, 100])
    print(theta[200, 200])
    G = np.asarray(G, np.uint8)
    cv2.imshow("p", G)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
