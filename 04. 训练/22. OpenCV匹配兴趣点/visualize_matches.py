import cv2


BINDINGS = {
    "SIFT": cv2.SIFT_create(),                  # 具有方向不变性和尺度不变性        普通描述子
    "ORB": cv2.ORB_create(),                    # 具有方向不变性和尺度不变性        二值描述子
    "BRISK": cv2.BRISK_create(),                # 具有方向不变性和尺度不变性        二值描述子
}


def select_matcher(optimize_method_name, descriptor_name, args=[]):
    if args and not isinstance(args, list):
        args = [args]
    norm_type = cv2.NORM_L2
    if descriptor_name in ["BRISK", "ORB"]:
        norm_type = cv2.NORM_HAMMING
    if optimize_method_name == "cv":                                # 交叉验证
        matcher = cv2.BFMatcher(norm_type, True).match
    elif optimize_method_name == "ratio":                           # 比率选择
        matcher = cv2.BFMatcher(norm_type, False).knnMatch                   
    elif optimize_method_name == "threashold":                      # 匹配差值的阈值化
        matcher = cv2.BFMatcher(norm_type, False).radiusMatch
    else:
        raise ValueError("Unknown optimize method: {}".format(optimize_method_name))
    return matcher, args   


def detect_and_visualize_keypoints(img1, img2, detector_name, descriptor_name, optimize_method_name, optimize_method_args=[], num_matches_to_visualize=10):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    detector = BINDINGS[detector_name]
    descriptor = BINDINGS[descriptor_name]
    kp1 = detector.detect(img1_gray)
    kp2 = detector.detect(img2_gray)
    kp1, des1 = descriptor.compute(img1_gray, kp1)
    kp2, des2 = descriptor.compute(img2_gray, kp2)
    matcher, args = select_matcher(optimize_method_name, descriptor_name, optimize_method_args)
    matches = matcher(des1, des2, *args)
    if optimize_method_name == "ratio":
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        matches = good
    selected_matches = matches[:num_matches_to_visualize]
    imageMatches = cv2.drawMatches(
        img1, kp1, img2, kp2, selected_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(f"/root/study/opencv/workspace/{detector_name}_{descriptor_name}_{optimize_method_name}.jpg", imageMatches)


if __name__ == "__main__":
    img1 = cv2.imread("/root/study/opencv/workspace/1.jpg", 1)
    img2 = cv2.imread("/root/study/opencv/workspace/2.jpg", 1)
    valid_detector_names = ["SIFT",  "BRISK"]
    valid_descriptor_names = ["SIFT", "BRISK"]
    valid_optimize_method_names = ["cv", "ratio"]
    
    for detector_name in valid_detector_names:
        for descriptor_name in valid_descriptor_names:
            for optimize_method_name in valid_optimize_method_names:
                if optimize_method_name == "ratio":
                    args = [2]
                elif optimize_method_name == "threashold":
                    args = [0.4]
                else:
                    args = []
                detect_and_visualize_keypoints(
                    img1=img1, 
                    img2=img2, 
                    detector_name=detector_name, 
                    descriptor_name=descriptor_name, 
                    optimize_method_name=optimize_method_name, 
                    optimize_method_args=args,
                    num_matches_to_visualize=10
                    )


