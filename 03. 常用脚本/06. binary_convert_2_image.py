import cv2
import numpy as np 


def binary_convert_2_image(imagePath):    
    with open(imagePath, 'rb') as file:
        binary_data = file.read()

    binary_data = np.frombuffer(binary_data, dtype=np.uint8)
    decoded_image = cv2.imdecode(binary_data, cv2.IMREAD_COLOR)
    cv2.imwrite("decoded_image.jpg", decoded_image)
        

if __name__ == "__main__":
    imagePath = "/root/SuperResolution/flask/images/org.jpg"
    binaryImage = binary_convert_2_image(imagePath)