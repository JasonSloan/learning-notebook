import cv2
import numpy as np
import os

if __name__ == '__main__':
    """使用OpenCV对图像进行透视变换校正"""
    path = "../data/croped/"
    imgs_names = os.listdir(path)
    for img_name in imgs_names:
        name = img_name.split(".")[0]
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
        # 二值化，TODO: 二值化的阈值需要调整
        thresh, binary = cv2.threshold(gray, thresh=80, maxval=255, type=cv2.THRESH_BINARY)
        # cv2.imwrite(f"{img_name}_binary.png", binary)
        # 寻找contour
        contours, result = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        # 计算contour的面积
        areas = [cv2.contourArea(contours[i]) for i in range(len(contours))]
        areas = np.array(areas)
        # 计算contour的面积占整个图像的比例
        ratio = areas / (img.shape[0] * img.shape[1])
        # 如果面积占比小于0.6，说明contour是无用的，就过滤掉
        ratio_mask = ratio > 0.6
        # 通过ratio_mask过滤掉无用的contour
        filtered_contours = [contours[i] for i in range(len(contours)) if ratio_mask[i]]
        if len(filtered_contours) == 0:
            print(f"{img_name} 没有足够的合适的contour")
            continue
        else:
            # 过滤ratio
            ratio = ratio[ratio_mask]
            # 获取ratio降序排列的排序索引
            sorted_indices = np.argsort(ratio)[::-1]
            if len(filtered_contours) == 1:
                # 如果只有一个contour，那么说明这个contour就是我们要找的
                contour = filtered_contours[sorted_indices[0]]
            elif len(filtered_contours) == 2:
                # 如果有两个contour，那么说明最大的contour是和整个图像大小一样的，我们要找的是第二大的contour（因为第二大的contour是我们要找的表盘）
                contour = filtered_contours[sorted_indices[1]]
        # 画出contour
        mask = np.zeros_like(img[:, :, 0])
        cv2.drawContours(mask, [contour], 0, (255), thickness=cv2.FILLED)
        cv2.imwrite(f"{img_name}_mask.png", mask)
        # 用mask来从原图中扣出表盘
        masked_image = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(f"{img_name}_masked.png", masked_image)

        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        # 获得最小外接矩形的四个顶点
        box = cv2.boxPoints(rect)

        # 指定旋转后的宽高
        width, height = 300, 300
        clockwise = True if rect[2] < 45 else False
        if clockwise:
            dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
        else:
            dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        # 构造透视变换矩阵
        M = cv2.getPerspectiveTransform(box, dst_pts)
        # 透视变换
        rotated_image = cv2.warpPerspective(masked_image, M, (width, height))
        cv2.imwrite(f"{img_name}_rotated.png", rotated_image)
