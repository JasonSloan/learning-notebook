
def convert_image_2_binary(imagePath):    
    with open(imagePath, "rb") as f:
        binaryImage = f.read()
    return binaryImage
        

if __name__ == "__main__":
    imagePath = "/root/SuperResolution/flask/images/org.jpg"
    binaryImage = convert_image_2_binary(imagePath)