import base64
import numpy as np
import cv2


def base64toArray(image_path):
    # Read the image
    with open(image_path, 'rb') as image_file:
        image_binary = image_file.read()
    # Encode the binary data to base64
    base64_encoded = base64.b64encode(image_binary).decode('utf-8')
    # Decode the base64 data into binary data
    image_binary = base64.b64decode(base64_encoded)
    # Convert the binary data to a NumPy array
    image_ndarray = np.frombuffer(image_binary, np.uint8)
    # Decode the NumPy array into an OpenCV image
    decoded_image = cv2.imdecode(image_ndarray, cv2.IMREAD_COLOR)
    cv2.imwrite('output_image_array.jpg', decoded_image)

if __name__ == "__main__":
    image_path = 'workspace/OST_009_croped_200_200.jpg'
    base64toArray(image_path)
