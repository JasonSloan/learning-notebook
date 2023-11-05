import base64
from PIL import Image
import io

def encode(img_path):
    with open(img_path, 'rb') as reader:
        s = reader.read()  # 二进制数据
        s = base64.b64encode(s)  # 图像的编码
        return s

def decode(s, img_path):
    s = base64.b64decode(s)  # 解码还原图像
    img = Image.open(io.BytesIO(s))
    # img.show()
    with open(img_path, 'wb') as writer:
        writer.write(s)
        
if __name__ == '__main__':
    # 编码
    es = encode('dog.jpg') 
    # 解码
    decode(es, 'dog_decode.jpg')
