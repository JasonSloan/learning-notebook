import cv2


def array_convert_2_binary(imageArray, height, width):    
    imageArray = imageArray.reshape(height, width, 3)
    # 如果编码成.png格式的数据就太大了，所以这里编码成.jpg格式的
    bytes_data = cv2.imencode(".jpg", recovered_image)[1].tobytes()
    # 验证一下将array转换回二进制是否正确
    with open("binaryImage.jpg", "wb") as f:
        f.write(bytes_data)
    return bytes_data
        

if __name__ == "__main__":
    imagePath = "/root/SuperResolution/flask/images/org.jpg"
    imageArray = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    height = imageArray.shape[0]
    width = imageArray.shape[1]
    imageArray = imageArray.ravel()
    binaryImage = array_convert_2_binary(imageArray, height, width)