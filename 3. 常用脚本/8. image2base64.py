import cv2
import base64


def image2base64(image_path):
    # Read the image
    with open(image_path, 'rb') as image_file:
        image_binary = image_file.read()
    # Encode the binary data to base64
    base64_encoded = base64.b64encode(image_binary).decode('utf-8')
    # Print or use the base64-encoded image data
    print(base64_encoded)
    with open("base64_encoded.txt", "w") as f:
        f.write(base64_encoded)

    #==========================================================================
    # Decode the base64 data into binary data
    image_binary = base64.b64decode(base64_encoded)
    # Specify the image file path where you want to save the decoded image
    output_image_path = 'output_image_base64.jpg'
    # Save the decoded image as a file
    with open(output_image_path, 'wb') as image_file:
        image_file.write(image_binary)
    print(f"Decoded image saved to {output_image_path}")

if __name__ == "__main__":
    image_path = 'workspace/OST_009_croped.jpg'
    image2base64(image_path)