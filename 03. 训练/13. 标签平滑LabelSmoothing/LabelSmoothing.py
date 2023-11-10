def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


if __name__ == '__main__':
    # cp: positive, cn: negative
    cp, cn = smooth_BCE(eps=0.95)

