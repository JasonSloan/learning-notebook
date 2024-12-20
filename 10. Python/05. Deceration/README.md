## 一、闭包

闭包的构成条件：

1，在函数嵌套（函数里面再定义函数）的前提下

2，内部函数使用了外部函数的变量（参数）

3，外部函数的返回值是内部函数的引用

```python
def outer(a):    
    b = 5    
    def inner():        
        c = 7        
        print(a + c)   
     return inner
```

## 二. 装饰器

（一），不带参数的函数装饰器

两层：外层参数为函数引用，内层函数为被引用函数的参数

```python
import time
def timer(func):
    def wrapper(sleep_time):
        t1 = time.time()
        func(sleep_time)
        t2 = time.time()
        cost_time = t2 - t1
        print(f"花费时间：{cost_time}秒")
    return wrapper

@timer  # 这个装饰器就是函数
def function(sleep_time):
    time.sleep(sleep_time)

function(3)
"""
执行顺序说明：
代码从上往下执行, 先导入time模块, 定义timer函数, 再执行
到@timer装饰器,
该函数装饰器没有调用, 再定义function函数, 当被装饰的函
数定义好了, 则将被
装饰的函数作为参数传入装饰器函数并执行, 即
timer(function), 返回wrapper函数的引用,
所以当最后执行function(3)时, 其实等价于wrapper(3), 而
wrapper函数中又调用了func函数,
即function函数 """
```

（二）带参数的函数装饰器

三层：最外层参数为装饰器的参数，次外层为函数的引用，最内层为被引用函数的参数

```python
import time


def interaction(name):
    def wrapper(func):
        def deco(work_time):
            # print("deco函数被调用")
            print(f"{name}, 你好, 请把要洗的衣物放进来!")
            func(work_time)
            print(f"{name}, 我帮你把衣物洗好了, 快来拿!")
        return deco
    return wrapper


@interaction("张三")
def washer(work_time):
    time.sleep(work_time)
    print("嘀嘀嘀...")


"""
    执行顺序说明：
    代码从上往下执行, 先导入time模块, 定义interaction函数,
    再执行到@interaction装饰器,
    该函数装饰器被调用, 则执行该函数, 定义wrapper函数, 并返
    回其引用, 再定义washer函数, 当
    被装饰的函数定义好了, 则将被装饰的函数作为参数传入刚刚执
    行装饰器返回的wrapper函数并执行,
    即wrapper(washer), 再定义deco函数, 并返回其引用, 所以
    当最后执行washer(3)时, 其实等价于
    deco(3), 而deco函数中又调用了func函数, 即washer函数
    """
washer(3)
```

（三）不带参数的类装饰器

```python
import time


class Timer:
    def __init__(self, func):
        self.func = func

    def __call__(self, sleep_time):
        t1 = time.time()
        self.func(sleep_time)
        t2 = time.time()
        cost_time = t2 - t1
        print(f"花费时间：{cost_time}秒")


@Timer  # 这个装饰器就是类
def function(sleep_time):
    time.sleep(sleep_time)


"""
执行顺序说明：
代码从上往下执行, 先导入time模块, 定义Timer类和方法, 执
行到@Timer装饰器时,
该类装饰器没有实例化, 再定义function函数, 当被装饰的函
数定义好了, 则将被
装饰的函数作为参数传入类装饰器并实例化, 即
Timer(function), 实例化调用初始化方法,
创建实例变量, __new__返回实例对象, 当最后执行
function(3)时, 其实等价于Timer(function)(3),
即调用__call__方法, 而该方法中又调用了func函数, 即
function函数 """
function(3)
```

（四）带参数的类装饰器

```python
import time
class Interaction:
    def __init__(self, name):
        self.name = name
    def __call__(self, func):
        def deco(work_time):
            print(f"{self.name}, 你好, 请把要洗的衣物放进来!")
            func(work_time)
            print(f"{self.name}, 我帮你把衣物洗好了, 快来拿!")
        return deco
@Interaction("张三")
def washer(work_time):
    time.sleep(work_time)
    print("嘀嘀嘀...")
"""
执行顺序说明：
代码从上往下执行, 先导入time模块, 定义Interaction类和
方法, 执行到@Interaction装饰器时,
该类装饰器进行实例化, 调用初始化方法, 创建实例变量,
__new__返回实例对象, 再定义washer函数,当被装饰的函数定义好了,
则将被装饰的函数作为参数调用实例对象, 即Interaction("张三")(washer),
即调用__call__方法, 定义deco函数并返回其引用, 所以当最
后执行washer(3)时, 其实等价于deco(3), 而其中又调用了
func函数, 即washer函数 """
washer(3)
```

