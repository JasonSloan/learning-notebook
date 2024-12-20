

class A:
    
    def __init__(self, x, y):
        self.x  = x
        self.y = y


class B:
    
    def __init__(self, x, y, z=None):
        self.x = x
        self.y = y
        self.z = z
        

if __name__ == "__main__":
    x = 1
    y = 2
    a_instance = A(x, y)
    b_instance = B.__new__(B)               # 只new出来B但是并没有初始化B
    # 假设a_instance有很多属性(比如要重写torch.Tensor), 又不想把a_instance中的属性值一个一个提取出来, 再赋值给b_instance
    # 那么就可以用下面的方式
    for k, v in vars(a_instance).items():   # 遍历A实例的所有属性，将其值赋值给B实例
        setattr(b_instance, k, v)
    print(b_instance.x)
    print(b_instance.y)
    