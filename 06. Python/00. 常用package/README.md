## 一、argparse（终端命令行添加参数）

```python
parser = argparse.ArgumentParser()   #用于解析命令行，将命令行的str解析为python中的对象
parser.add_argument('--epochs', type=int, default=30)
opt = parser.parse_args()
# parse_known_args() ：返回一个元祖，第一个元素是已知命名的参数，第二个是未知命名的参数
opt.epochs
#parser.add_argument()内可以指定的参数：
# default，即这个参数的默认值。
# type，参数的数据类型，例如int, float, str, list等。
# required，默认为false，如果为true，并且没有默认值，表示这个参数用户必须指定，否则程序报错。
# action，action =‘store_true’ / ‘store_false’。使用这个选项的参数必须为布尔变量。其中store_true表示：用户指定了这个参数，那么这个参数就为true。否则为False，反之亦然。
# choice，即参数的取值范围，可以帮你自动判断是否越界。例如[1, 2]，表示只可以取值1或者2。
# help，参数的解释，相当于是注释作用，怕自己忘了。
# nargs，取值可以是[“ * ”, “+”, 正整数]等，就是说这个参数是一个列表，"*" 表示任意个参数。“+”表示至少一个参数。正整数表示这么多个参数。假设这个参数名字叫做people，且nargs = 2。那么命令如下：python main.py - -people x y。那么people = [“x”, “y”]。
```

## 二、glob（查找文件路径）

```python
glob.glob('./**/' + file, recursive=True)   #recursive为true的时候，**可以匹配所有文件夹
glob.glob('./train/*.jpg' )      #查找train文件夹下所有的.jpg文件
```

## 三、yaml（将yaml文件解析为一个字典）

```python
安装：pip install PyYAML
with open('./cfg/hyp.yaml') as f:
    hyp = yaml.safe_load(f)
#其中hyp.yaml文件的存储格式为
#key: value
with open('./cfg/hyp.yaml', "w") as fp:
    yaml.safe_dump(label_dic, fp)
```

## 四、datetime

```python
datetime.datetime.now().strftime("%Y%m%d-%H%M%S")    #返回年月日-时分秒
```

## 五、numel()   统计模型参数数量

```python
a = torch.randn(2,2,3)
print(a.numel())    #结果为12
num_params = sum(param.numel() for param in net.parameters())
```

## 六、thop和torchstat（计算展示模型中的总参数以及计算量）

```python
# torchstat的缺点：  
# 1. 限制模型输入仅能为图片
# 2. 限制模型每一个layer的输入须为单个变量，即不能算一个batch的计算量
from torchstat import stat
import torchvision.models as models

model = models.vgg16()
stat(model, (3, 224, 224))  #第一个参数传模型，第二个参数传输入的图片的维度

# thop包
from thop import profile
from thop import clever_format

input = torch.randn(1, 3, 224, 224)  #注意这个包要传入的是一个batch的真正的图片，而不是图片的维度
flops, params = profile(model, inputs=(input, ))
print(flops, params) # 1819066368.0 11689512.0
flops, params = clever_format([flops, params], "%.3f") #将参数包装一个更好看的格式
print(flops, params) # 1.819G 11.690M
```

## 七、shutil移动文件到指定路径

```python
 import shutil
 shutil.rmtree()   # 删除文件夹，os和os.path只能删除空文件夹
 for each_path in imgs_path:
     # （相当于linux的mv)
     shutil.move(each_path,'./train/imgs')
# 拷贝文件a.txt到./data下，相当于cp操作
shutil.copy2('a.txt', './data')     
```

## 八、tqdm（可视化进度条显示耗时进程）

```python
用法一：
from tqdm import tqdm
import time
l = [i for i in range(100)]
for i in tqdm(l):
    time.sleep(1)
    
用法二：
labels_generator = tqdm(labels_list)
for i in labels_generator:
    labels_generator.desc = 'Caching labels(%g found,%g missing,' \
                                  ' %g empty, %g duplicated,for %g images)'\
                                  %(number_of_found, number_of_missing,
                                    number_of_empty, number_of_duplicate,number_of_imgs)
                                
用法三：
image_path_list = tqdm(img_path_list, desc="Reading image shapes")
for f in image_path_list：
    Image.open(f)   #会自动显示img_path_list中的数量和当前读取的动态变化的数量
    
训练代码中用法:
from tqdm import tqdm 
import time 
epochs = 100
loss=13 
for epoch in range(epochs): 
    pbar = tqdm(enumerate(dataloader),total=nb)    # 要加total否则不会显示剩余估计时间
    for batch in pbar:
        time.sleep(0.1) 
        loss -= .002 
        pbar.set_description(f"epoch:{epoch}/{epochs-1},batch:{batch}, Loss: {loss:.2f}") 
pbar.close()
```

## 九、xmltodict（解析xml文件用）

```python
import xmltodict
with open(r'E:\AI\pytorch\YOLO\yolov3_spp\data\VOCdevkit\VOC2012\Annotations\2007_000027.xml') as fp:
    xml_content = fp.read()
    xml_dict = xmltodict.parse(xml_content)
```

## 十、vars（将一个类的实例对象的属性名与属性值组成字典返回）

```python
class ni():
    def __init__(self, a, b):
        self.c = a
        self.d = b
wo = ni(2, 4)
print(vars(wo))
>>{'c': 2, 'd': 4}
```

## 十一、subprocess（在py文件中调用终端在终端执行命令）

```python
import subprocess
cmd = "python testttttttttttt.py" #指定命令
# cmd = "pip install lxml"
output = subprocess.check_output(cmd, shell=False) #在终端中执行命令，shell=False代表的是命令之间用空格分开
output = output.decode() #返回值是byte类型，所以要decode一下转换成字符串
print(output)
```

## 十二、pathlib

from pathlib import path

| Path.cwd()                               | 获取当前目录                                   |                                          |
| ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| Path.home()                              | 获取Home目录                                 |                                          |
| Path.cwd().parent                        | 获取上级父目录                                  |                                          |
| example_path = Path('/Users/Anders/Documents/abc.gif') |                                          |                                          |
| example_paths.suffix                     | 返回目录中多个扩展名列表                             | .gif                                     |
| example_paths.suffixes                   | 返回目录中多个扩展名列表                             | ['.tar', '.gz']                          |
| example_path.stem                        | 返回目录中最后一个部分的文件名                          | abc                                      |
| example_path.name                        | 返回目录中最后一个部分的文件名                          | abc.gif                                  |
| example_path.with_name('def.gif')        | 替换目录最后一个部分的文件名并返回一个新的路径                  | /Users/Anders/Documents/def.gif          |
| example_path.with_suffix('.txt')         | 替换目录最后一个部分的文件名并返回一个新的路径                  | /Users/Anders/Documents/abc.txt          |
| Path('/Users/Anders/Documents/').joinpath('python_learn') | 路径拼接                                     |                                          |
| example_path = Path('/Users/Anders/Documents')[path for path in example_path.iterdir()] | 遍历文件夹                                    | [PosixPath('/Users/Anders/Documents/abc.jpg'),PosixPath('/Users/Anders/Documents/book-master'),PosixPath('/Users/Anders/Documents/Database'),PosixPath('/Users/Anders/Documents/Git'),PosixPath('/Users/Anders/Documents/AppProjects')] |
| example_path = Path('/Users/Anders/Documents/test1/test2/test3') |                                          |                                          |
| example_path.mkdir(parents = True, exist_ok = True) | 创建文件夹(parents：如果父目录不存在，是否创建父目录。exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。） |                                          |
| example_path.rmdir()                     | 删除路径对象目录                                 |                                          |
| .is_dir()                                | 是否是目录                                    |                                          |
| .is_file()                               | 是否是普通文件                                  |                                          |
| .resolve()                               | 返回一个新的路径，这个新路径就是当前Path对象的绝对路径，如果是软链接则直接被解析 |                                          |
| .exists()                                | 该路径是否指向现有的目录或文件                          |                                          |
| FILE = Path(__file__).resolve()  FILE.parents[0].as_posix() | #第一句直接返回当前文件所在位置的绝对路径#第二句parents是返回父路径，as_posix()是将路径中的反斜杠\替换为正斜杠/ |                                          |
| Path().resolve()                         | 返回工作目录，与Path(__file__).resolve() 区别是后者返回的是该代码所在文件的目录，前者返回的是入口文件的工作目录 |                                          |
| Path(path).stat().st_mtime               | 返回path文件的修改时间                            |                                          |

## 十三、os和os.path

```python
os.getcwd()返回当前工作目录
os.chdir(path)改变当前目录
os.listdir(path)返回 path 指定的文件夹包含的文件或文件夹的名字的列表
os.mkdir(path)创建单层目录
os.makedirs(path, exist_ok=True)创建多层目录,exist_ok=True意思是如果目录已存在,则相当于什么都不做
os.rmdir()删除 path 指定的最后一级空目录，如果目录不存在或不为空，都会引发异常
os.remove(path)移除 path 对应的文件；如果 path 是目录或者文件不存在，都会引发异常

os.removedirs(path)递归删除空目录
os.curdir()指代当前目录（“.”）
os.pardir()指代上一级目录（“..”）
os.rename(src, dst)重命名目录或文件
os.renames(old, new)递归重命名目录或文件，会自动创建新路径所需的中间目录
os.cpu_count()   返回当前设备最大支持线程数
```

```python
os.path.abspath(path)返回路径 path 的绝对路径
os.path.basename(path)返回路径 path 最后一级的名称，通常用来返回文件名

os.path.dirname(path)返回路径 path 的目录名称
os.path.split(path)把路径分割成 dirname 和 basename，返回一个元组
os.path.splitext(path)把路径中的扩展名分割出来，返回一个元组
os.path.exists(path)path 路径存在则返回 True，不存在或失效则返回 False
os.path.isabs(path)判断 path 是否是绝对路径，返回 True 或 False
os.path.isfile(path)判断路径是否为文件，返回 True 或 False
os.path.isdir(path)判断路径是否为目录，返回 True 或 False
os.path.join(path, *paths)智能地拼接一个或多个路径部分
os.path.splitext('c:\test.csv')
'c:\test','csv'
os.path.basename('c:\test.csv')
'test.csv'
os.path.basename('c:\csv')
'csv' （这里csv被当作文件名处理了）
os.path.basename('c:\csv\')
返回空
# 获取当前工作文件的绝对路径
os.path.dirname(os.path.abspath(__file__))
# 将linux中的家目录(即~)替换成绝对路径
os.path.expanduser("~/data/")
'D:\\AI\\pytorch/data'
```

##  十四、json

```python
json.load：表示读取文件，返回python对象
json.dump：表示写入文件，文件为json字符串格式，无返回
json.dumps：将python中的字典类型转换为字符串类型，返回json字符串 [dict→str]
json.loads：将json字符串转换为字典类型，返回python对象 [str→dict]
load和dump处理的主要是 文件
loads和dumps处理的是 字符串

1,dumps和loads
import json
info = {'name': 'Tom', 'age': 18, 1: 'one'}
print(type(info), info)
with open("info.json", mode="w") as f:
    info_str = json.dumps(info,ensure_ascii=False) # 序列化
    print(type(info_str), info_str)
    f.write(info_str)
with open("info.json") as f:
    content = f.read()
    print(type(content), content)
    res = json.loads(content) # 反序列化
    print(type(res), res)


2,dump和load
import json
info = {'name': 'Tom', 'age': 18, 1: 'one'}
with open("info.json", mode="w") as f:
    json.dump(info, f)
with open("info.json") as f:
    res = json.load(f)
    print(type(res), res)



import json
with open('./TestFile.json') as file:
    content = json.load(file)
obj_list = content["outputs"]["object"]
for item in obj_list:
    print(item["name"])
    print(item["bndbox"])
```

## 十五、pickle

```python
import pickle
info = {'name': 'Tom', 'age': 18, 1: 'one'}

with open("info.pickle", mode="wb") as f:
    info_byte = pickle.dumps(info)
    print(type(info_byte), info_byte)
    f.write(info_byte)
with open("info.pickle", mode="rb") as f:
    content = f.read()
    print(type(content), content)
    res = pickle.loads(content)


import pickle
info = {'name': 'Tom', 'age': 18, 1: 'one'}
with open("info.pickle", mode="wb") as f:
    pickle.dump(info, f)
with open("info.pickle", mode="rb") as f:
    res = pickle.load(f)
    print(type(res), res)
```

## 十六、sys

```python
sys.path # sys 模块的 path 变量包含了 Python 解释器自动查找所需模块的路径的列表。如果需为系统搜索路径添加新路径，在每个工程文件中只需要添加一次就行，添加一次，整个工程中的py文件都会拥有一样的搜索路径。
sys.argv[0]  # 返回当前主入口执行文件所在路径
```

## 十七、graphviz(决策树可视化)

```python
下载安装包(msi安装包): http://www.graphviz.org/
执行下载好的安装包(双击msi安装包)

# 执行以下代码
import pydotplus
from sklearn import tree
import os
# 将graphviz的bin目录添加到搜索路径中
os.environ["PATH"] += os.pathsep + r'D:\installed_software_dir\graphviz\bin'
# 其中algo为已训练好的决策树对象，feature_names为特征名，class_names为类名（在本例子中共有3类）
dot_data = tree.export_graphviz(decision_tree=algo, out_file=None,
                                feature_names=['A1', 'B1', 'C1', 'D1'],
                                class_names=['a', 'b', 'c'],
                                filled=True, rounded=True,
                                special_characters=True,
                                node_ids=True
                                )
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("iris03.png")
graph.write_pdf("iris04.pdf")
```

## 十八、joblib（机器学习用来保存模型(sklearn)的包）

```python
# 保存模型
joblib.dump(model,"./model.m")
# 加载模型
model = joblib.load(model_path)
```

## 十九、base64（图片编码传输）

```python
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
    img.show()
    with open(img_path, 'wb') as writer:
        writer.write(s)
        
if __name__ == '__main__':
    # 编码
    es = encode('../datas/飞机.jpg') 
    # 解码
    decode(es, './小狗.png')
```

## 二十、sklearn.model_selection中的KFold

```python
def kFold_cv(X, y, classifier, **kwargs):
    """
    :param X: 特征
    :param y: 目标变量
    :param classifier: 分类器
    :param **kwargs: 参数
    :return: 预测结果
    """
    kf = KFold(n_splits=5, shuffle=True) # 5折交叉验证
    y_pred = np.zeros(len(y))    # 初始化y_pred数组
    start = time.time()
    for train_index, test_index in kf.split(X):  
        X_train = X[train_index]    
        X_test = X[test_index]
        y_train = y[train_index]    # 划分数据集
        clf = classifier(**kwargs)    
        clf.fit(X_train, y_train)    # 模型训练
        y_pred[test_index] = clf.predict(X_test)    # 模型预测
    print("used time : {}".format(time.time()-start))
    return y_pred  
```

## 二十一、queue(先入后出)

```python
from queue import LifoQueue

q = LifoQueue()
q.put(1)
q.put(2)
q.put(3)
for i in range(3):
    print(q.get())
q.empty()
```

## 二十二、faiss（向量索引库，速度极快）

```python
Note:使用时，所有数据精度必须是float32，否则会报错
参考：https://zhuanlan.zhihu.com/p/357414033，已收藏
# 安装：
conda install faiss-cpu -c pytorch（必须使用conda安装）
# 构建索引向量库：
index = faiss.index_factory(dim, 'HNSW32', faiss.METRIC_INNER_PRODUCT)  # dim为向量维度，HNSW32为构建向量库的方式（检索方式），faiss.METRIC_L2为查找相似向量的时候使用什么方法度量距离
# 向index索引向量库中添加向量    
index.add(vec)  
# 在index向量库中查找最近的k个向量，返回这k个向量的距离以及索引值
distance, _index = index.search(_vec, k)  

# 构建向量库的方式（检索方式）
Flat ：暴力检索
HNSWx ：图检索（HNSWx中的x为构建图时每个点最多连接多少个节点，x越大，构图越复杂，查询越精确）

# faiss包括的度量方法包括：
    .METRIC_INNER_PRODUCT（内积）（余弦相似度）（注意使用此方式进行向量检索的时候需要将向量进行归一化）
    .METRIC_L1（曼哈顿距离）
    .METRIC_L2（欧氏距离）
    .METRIC_Linf（无穷范数）
    .METRIC_Lp（p范数）
    .METRIC_BrayCurtis（BC相异度）
    .METRIC_Canberra（兰氏距离/堪培拉距离）
    .METRIC_JensenShannon（JS散度）
```

## 二十三、labelImg（图片标注）

```python
安装：anaconda中：pip install labelImg
使用：在conda中输入labelImg命令唤起GUI界面
快捷方式：
    Ctrl + s  保存标注
    w         画外接矩形
    d         切换到下一张图片
    a         切换到上一张图片
    del       删除矩形框
```

## 二十四、anaconda

```python
conda create -n pytorch python=3.6    #创建环境-n(name)代表环境名字叫pytorch
conda remove -n pytorch --all         #删除环境-n(name)代表环境名字叫pytorch

conda create -p /root/workspace/pai/ENV python=3.7  #Linux上创建conda环境
conda activate /root/workspace/pai/ENV    #Linux上激活conda环境
cd /root/workspace/pai                #进入目标文件夹
ENV/bin/pip install torch==1.10.2 cpuonly#Linux上安装包


conda env list          #查看所有环境
conda activate pytorch           # 进入名字为pytorch的环境
conda deactivate    #退出当前环境
#按照pytorch官网安装的命令复制下来安装pytorch
 python 
 import torch      #检验pytorch是否安装成功
 torch.cuda.is_available()    #检验GPU是否可用
 
 
#pycharm使用anaconda创建的环境：
#在pycharm中新建project，选择existing environment，选择conda environment，选择已经在anaconda  prompt中已经创建好的环境pytorch目录下的python.exe。

                
#安装Jupyter，在conda prompt中安装
conda activate pytorch           # 进入名字为pytorch的环境
conda install nb_conda       #安装包
jupyter notebook          #进入jupyter notebook
shfit+enter              #进入到Jupyter后，新建代码块，按shfit+enter可以执行该行并切换到下一行


#检验GPU是够能使用cuda
#在任务管理器中的性能中查看GPU型号，到https://www.nvidia.cn/geforce/technologies/cuda/supported-gpus/网址上查看GPU是否支持cuda
```

## 二十五、gc（python中的垃圾回收模块）

```python
gc.collect()   #回收内存中的垃圾（即引用次数为0的变量）
```

## 二十六、pygwalker（机器学习数据可视化）

```python
需要再jupyter notebook中使用
import pandas as pd
import pygwalker as pyg
data_path = "joint_data_from_ycsp221007_and_ycsp221007_v.csv"
df = pd.read_csv(data_path)
gwalker = pyg.walk(df,hideDataSourceConfig=True, vegaTheme="vega")
```

## 二十七、lap（线性分配）（类似于动态规划）

```python
# 什么是线性分配问题可以这么问chatgpt：give  me  an vivid actual example  of linear assignment

import numpy as np
import lap

# 代价矩阵
cost_matrix = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])

# 计算最优匹配方式（目标是使所有的i和所有的j匹配上，但是使总cost最小）
cost, row_ind, col_ind = lap.lapjv(cost_matrix)

# Print the optimal assignments
for i in range(len(row_ind)):
    print(f"Row {i} is assigned to Column {col_ind[i]}")

```

## 二十八、netron网络可视化

```python
import netron
netron.start("yolov3.pth")
```



