[对axis的理解链接](https://blog.csdn.net/mahoon411/article/details/114777623?spm=1001.2014.3001.5506)

python中的广播机制：沿着最后一个维度对齐，然后广播

# 0、一些设置

```python
# 设置不使用科学计数法  #为了直观的显示数字，不采用科学计数法
np.set_printoptions(precision=3, suppress=True)
```

# 一、数组属性

```python
.shape     #返回矩阵a的维度
.ndim      #返回矩阵a的维度的维度
.size      #返回矩阵a的元素总个数
.dtype     #返回矩阵a中元素的类别
.itemsize  #返回矩阵a中每个元素的大小，以字节为单位
```

# 二、数组创建

## 1，从现有的数据创建

```python
np.array([(1,2,3),(4,5,6)],dtype=np.float32)  %创建矩阵，必须传入列表,2*3。可以指定每个元素的类型为float或者int
np.asarray(a, dtype=None) # 类似于 np.array，主要区别是当 a是 ndarray 且 dtype 也匹配时，np.asarray不执行复制操作，而 np.array 仍然会复制出一个副本，占用新的内存
np.fromiter(iterable, dtype, count=-1 # 从可迭代对象创建一个新的一维数组并返回
 d = a.copy()         #深拷贝，完全拷贝
```

## 2，从形状或值创建

```python
np.empty((2, 3))             # 返回给定形状和类型且未初始化的新数组
np.empty_like(prototype, dtype=None) # 返回形状和类型与给定 prototype 相同的新数组
np.zeros((3, 4))             # 创建元素全部为0的矩阵，3*4，
np.zeros_like(X)             #创建一个矩阵，维度和X相同，元素全部为0
np.ones((2, 3, 4, 5))         #创建元素全部为1的矩阵，2*3*4*5，2行3列，每个元素有4行5列个子元素
np.ones_like(X)              #创建一个矩阵，维度和X相同，元素全部为1
np.full(shape, fill_value, dtype=None) # 返回给定形状和类型的新数组，并用 fill_value 填充
np.full_like(a, fill_value, dtype=None) # 返回一个与给定 a 具有相同形状和类型的 fill_value 填充的数组
np.eye(N, M=None, k=0, dtype=np.float64) #返回对角线为1，其他地方为0（单位矩阵）
np.identity(n, dtype=np.float64) # 返回 n*n 的单位数组（主对角线为1，其他元素为0的方形数组）
```

## 3，从数值范围创建数组

```python
np.arange(10, 30, 5)         %创建矩阵，从10开始到30结束，步长为5，左开右闭，包括10不包括30
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)      #创建等差矩阵
numpy.logspace(start, stop, num=50, endpoint=True, base=10.0,dtype=None) # 创建等比矩阵
```

# 三、dtype常用值

```python
np.int8 字节（-128 to 127）
np.int16 整数（-32768 to 32767）
np.int32 整数（-2147483648 to 2147483647）
np.int64 整数（-9223372036854775808 to 9223372036854775807）
np.uint8 无符号整数（0 to 255）
np.uint16 无符号整数（0 to 65535）
np.uint32 无符号整数（0 to 4294967295）
np.uint64 无符号整数（0 to 18446744073709551615）
np.float16 半精度浮点数
np.float32 单精度浮点数
np.float64 双精度浮点数
```

# 四、矩阵运算

## １、matmul、@、mat、I（逆）

```python
A * B             %矩阵A的每个元素与矩阵B一一对应相乘
A @ B             %矩阵A的与矩阵B相乘，矩阵相乘
np.matmul(a, b)   %矩阵A的与矩阵B相乘，矩阵相乘
a = np.mat(a)         %转成矩阵形式，此时矩阵运算就是*而不是@，矩阵的逆a.I
```

## 2、sin、cos

```python
np.sin(a)         %将a矩阵进行sin运算，（注意必须使用弧度值）
np.cos()
np.tan()
np.arcsin()
np.arccos()
np.arctan()
```

## 3、floor、ceil

```python
np.floor()        # 返回 x 的底限
np.ceil()         # 返回 x 的上限
```

## 4、max、min、argmax、argmin、maximum、minimum、var、mean、std、media（中位数）、all

```python
.max(axis=None, keepdims=False)  # （一个数组内元素比较）返回沿给定轴的最大值，axis没有指定时，默认为None，表示返回所有元素的最大值
np.argmax(a, axis=None)          # 返回沿轴的最大值的索引
np.maximum(x1, x2)               # （两个数组进行比较）返回x1和x2逐个元素比较中的最大值
.min(axis=None, keepdims=False)  # 返回沿给定轴的最大值，axis没有指定时，默认为None，表示返回所有元素的最大值 
np.argmin(a, axis=None)          # 返回沿轴的最小值的索引
np.minimum(x1, x2)               # 返回x1和x2逐个元素比较中的最小值
.mean(axis=None, keepdims=False) # 返回沿给定轴的平均值，axis没有指定时，默认为None，表示返回所有元素的平均值
.var(axis=None, keepdims=False)  # 返回沿给定轴的方差，axis没有指定时，默认为None，表示返回所有元素的方差
.std(axis=None, keepdims=False)  # 返回沿给定轴的标准差，axis没有指定时，默认为None，表示返回所有元素的标准差
np.media(a)                      # 求中位数
a.all()                          # 如果a中全部元素为True，那么就返回True，否则返回False
```

## 5、prod（连乘）、sum

```python
.prod(a, axis=None, keepdims=np._NoValue, initial=np._NoValue)  # 返回给定轴上数组元素的乘积
.sum(a, axis=None, keepdims=np._NoValue, initial=np._NoValue)   #　返回给定轴上数组元素的和
```

## 6、e、exp、log、log2、log10

```python
np.e              # e的值
np.exp(a)         %将a矩阵进行e的对数运算
np.log(x)　　　　　# 计算 x 的自然对数
np.log2(x)　　　　# 计算 x 的以 2 为底的对数
np.log10(x)       # 计算 x 的以 10 为底的对数
```

## 7、sort、argsort

```python
np.sort(a, axis=-1))  # 返回排序之后的新数组
np.argsort(a)         # 返回排序之后的新数组的索引
```

## 8、nonzero、where、argwhere、unique

```python
np.nonzero(a)  　　　#　返回非零元素的索引
np.where(condition, x=None, y=None)　　# 当where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式;当where内有三个参数时，第一个参数表示条件，当条件成立时where方法返回x，当条件不成立时where返回y。condition : array_like, bool ，x, y：array_like，都么都传参，要么都不传；如果传三个参数，条件成立返回x，不成立时返回y；如果只传第一个参数，返回符合条件的元素的索引
np.argwhere(a)      # 找出数组中按元素分组的非零元素的索引
np.unique(ar, return_index=False, return_inverse=False,return_counts=False,axis=None)              # ar：输入的数组return_index：为True时，还会返回新数组元素在旧数组中的下标return_inverse：为True时，还会返回旧数组元素在新数组中的下标 return_counts：为True时，还会返回去重数组中的元素在原数组中的出现次数axis：指定操作的轴。没指定时，默认扁平化处理返回数组中排序的唯一元素，重复元素会被去除，只保留一个
```

## 9、bincount（计算数组中每个非负整数出现的次数）

```python
np.bincount(classes, minlength=nc)   # 用于计算给定数组中每个非负整数出现的次数
```

# 五、矩阵索引和切片

## 1，数组切片不会复制内部数组数据，只会生成原始数据的新视图

## 2，数组支持多维数组的多维索引和切片

```python
###多维数组的多维索引和切片案例
x = np.array([[1, 2], [3, 4], [5, 6]])
print(x[[1, 2]]) # 等价于x[1]和x[2]组成的数组
print(x[[0, 1, 2], [0, 1, 0]]) # 等价于x[0, 0]、x[1, 1]和x[2, 0]组成的数组
print(x[[True, False, True]]) # 等价于用bool值判断对应位置的数组保留与否
print(x[x < 4])  # 获取x中小于4的数据
```

# 七、矩阵维度转换，矩阵的拼接、拆分

## 1，reshape、resize

```python
#将矩阵拉直，以下三种方法其实都一样
a.ravel()        a.flatten()       a.reshape(-1)    
a.reshape((3,4))       #返回3*4矩阵
a.reshape((3,-1))      #返回一个新矩阵，维度维度为3*？，另一个维度会被自动计算出来
a.resize((2, 6))       #无返回值，是inplace操作
```

## 2、swapaxes、transpose

```python
np.swapaxes() #交换数组的两个轴
np.transpose   # 通过axes参数排列数组的shape，如果axes省略，则transpose(a).shape == a.shape[::-1] (同reshape不一样，转置是行变列，列变行。reshape是先拉直矩阵再按照拉直后的顺序变换成所要维度)
.T                    #转置矩阵
```

## 3、expand_dims、squeeze

```python
np.expand_dims(a, axis)     #扩展数组的形状
np.squeeze(a, axis=None)    #从给定数组的形状中删除维度为1的条目
a[:, np.newaxis]       #增维：对于n维矩阵a，转化成(n+1)维矩阵，转化为列向量
a[np.newaxis, :]       #增维：对于n维矩阵a，转化成(n+1)维矩阵，转化为行向量 
```

## 4、concatenate、stack、tile、repeat

```python
np.concatenate((a1, a2, ...), axis=0)    # 沿现有轴连接一系列数组，如果axis为None，则数组在使用前会被扁平化  （拼接，拼接后不增加维度）
np.stack(arrays, axis=0)  # arrays：Sequence[ArrayLike]  沿新轴连接一系列数组，最常用就是axis等于0  （堆叠，堆叠后增加一个维度）
np.vstack((a, b))      #纵向堆叠矩阵a和b，保持列不变
np.hstack((a, b))      #横向堆叠矩阵a和b，保持行不变
np.repeat(a, repeats, axis=None)  # 重复数组中的元素，axis：指定重复值的轴。没指定时，默认扁平化处理
np.tile(a, 20)          #将矩阵a沿着1轴的方向拼接20次
```

## 6、append、insert、delete

```python
np.append(a, b, axis=0)  #将矩阵（列表）b沿着0轴方向添加到a矩阵中，其中b和a的维度数必须相同，且b除了axis的维度其余维必须与a相同。（具体可以参考函数文档）
np.insert(a, 3, b, axis=0)  #将矩阵（列表）b沿着0轴方向在索引位置为3处插入到a矩阵中，其中b维度比a维度小1维。（具体可以参考函数文档）
np.delete(a, 2, axis=0) #删除矩阵a中沿着第0维度的索引为2（第2行）的元素
```

# 八、矩阵的复制和视图

```python
 b = a                #不拷贝，b和a是同一个矩阵的两个名字，指向同一个对象通过b改变矩阵，用a得到矩阵也是变         
 c = a.view()         #浅拷贝，创建了一个新对象c，但是c和a指向的矩阵数据是同一个，改变c的维度，a维度不变；但是改变c的值，a也变
 d = a.copy()         #深拷贝，完全拷贝，拷贝出来的b和a的id值不一样，此时a的值改变，b的值不变
```

# 九、文件写入读取

```python
np.save("arr.npy", arr, allow_pickle=True)     # 将矩阵arr写入文件arr.npy
np.load(file)       # Load arrays or pickled objects from ``.npy``, ``.npz`` or pickled files
```

