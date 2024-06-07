# 〇、pandas中设置显示全部

```python
# 显示所有列(参数设置为None代表显示所有行，也可以自行设置数字)
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置数据的显示长度，默认为50
pd.set_option('max_colwidth',200)
# 禁止自动换行(设置为Flase不自动换行，True反之)
pd.set_option('expand_frame_repr', False)
# 设置不使用科学计数法  #为了直观的显示数字，不采用科学计数法
pd.set_option('display.float_format', lambda x: '%.2f' % x)
```

# 一、创建Series、DataFrame

```python
#Series使用方法类似于字典
pd.Series(data=None, index=None, dtype=None, name=None)  #data：array-like, dict, or scalar value；index：必须是不可变数据类型，允许相同。不指定时，默认为从 0 开始依次递增的整数；

pd.DataFrame(data=None, index=None, columns=None, dtype=None) # data：array-like, dict, or DataFrame；index：行索引。不指定时，默认为从 0 开始依次递增的整数；columns：列索引。不指定时，默认为从 0 开始依次递增的整数
#通过字典嵌套series（列表）创建dataframe，字典中的key相当于每一列的列名
data = {"name":pd.Series(["张三","李四","王五"]),
        "age":pd.Series([13,15,12])}
df = pd.DataFrame(data)
```

# 二、Series、DataFrame属性

```python
.dtype   # 数据类型
.index   # 索引
.name   #  对象名称
.shape   
.ndim   
.size   # 元素数量
.dtypes   #会返回每一个字段（列）的数据类型，如果该列中有各种数据类型，那么数据类型就为object
.index   #行名
.columns #列名（字段名）
.axes    # 返回一个列表，包含行名和列名
.values  # 返回df对象的值，返回的值为ndarray格式
.列名    # 返回指定列名的值
df["name"]  # 类似于字典的方式取值，name为其中一列的字段名
```

# 三、DataFrame的转置

```python
.T   # 转置，行变列，列变行
```

# 四、Series、DataFrame内置方法

```python
.head(n=2)   #查看头两行
.tail(n=2)   # 查看尾两行
.map({"No":0,"Yes":1})    # 只有series的方法，非inplace操作：将No值映射为0，Yes值映射为1(一般用于处理y值那一列)
.replace('?', np.nan, inplace=True)   # 替换，一般用于处理缺失值
.drop(["某字段"], axis=1)     # 删除某字段 
.dropna()      # 去掉np.nan值
.nuique()     # 可以判断某个字段如果unique的值如果小于1，那么证明在该字段所有数据的值全一样，那就可以删除了
.isnull()   # 检测缺失值
.notnull()  # 检测非缺失值
.info()      # 打印df对象包含的数据信息（列名，每一列非空数据的总数，每一列的数据类型，占用内存）
.describe()  # 针对数值型的列，返回统计信息（总共多少数据、均值，标准差、最小值、最大值）
.shift(periods=2,axis=1,fill_value=3) #沿着1轴（列方向）移动2步，空出来的值用3填充
.reset_index(drop=True,inplace=True) #重置行索引，一般用在concat前一步
```

# 五、df中的增删改查(insert、pop、append）

```python
df["height"] = [170,165,180]  # 增加一列身高，值为170,165,180
df.insert(loc, column, value)  # 插入一列，loc：int，整数列索引，指定插入数据列的位置，column：新插入的数据列的名字，value：int, Series, or array-like，插入的数据
df.pop(item)  # item：要删除的列标签,inplace操作
df.drop(item,axis=1)  # item：要删除的标签，axis指定为行标签还是列标签，非inplace操作
.append(other, ignore_index=False) # 相当于numpy中的concatenate操作
```

#  六、对数据进行索引、操作行和列

```python
##################################################################
索引索引索引索引索引索引索引索引索引索引索引索引索引索引索引索引索引索引

# 1，直接索引
df["name"]     # 按列索引，取出字段名为name的那一列的值
df[["name","age"]] # 按列索引，取出字段名为name和age的两列


# 2， 标签索引
df.loc[["name","age"]] # 按行索引，取出字段名为name和age的两行
df.loc["name":"age"]   # 切片，取出标签为"name"到"age"之间的行
df.loc["row1","name"]  # 取出标签为"row1"行"name"列的值

# 3， 下标索引
.iloc[0,0]   # 取出df对象中位置为第一行第一列的元素（类似于列表）

索引索引索引索引索引索引索引索引索引索引索引索引索引索引索引索引索引索引
###################################################################

操作数据操作数据操作数据操作数据操作数据操作数据操作数据操作数据操作数据
# 按照行和列对数据进行操作
.apply(fun, axis=0)  #对数据的第0维度按照fun函数（fun为自己定义的函数）的方式进行处理并返回（类似于map)，注意使用fun作用完的每个元素值为str类型
# 按照每个元素对数据进行操作
.pipe(lambda x:x+2)   #对每个元素进行+2操作
.dropna()   # 去掉有缺省值的数据（按行去除）
.drop_duplicates()  # 删除重复值（按行去除）
.get_dummies(X, columns=["x_3", "x_4"]) # 指定对列名为"x_3", "x_4"的列进行哑编码（one-hot)
```

#  七、重置索引

```python
Series.reindex(index=None, fill_value=np.NaN)
index：要获取数据的索引，传入列表
返回重新索引组成的新的 Series 对象
```

#  八、Series重索引（按照新索引与原索引对应的值形成新的Series）

```python
Series.reindex(index=None, fill_value=np.NaN)
index：要获取数据的索引，传入列表
返回重新索引组成的新的 Series 对象
```

#  九、pandas中的统计数据分析

```python
.info()      # 打印df对象包含的数据信息（列名，每一列非空数据的总数，每一列的数据类型，占用内存）
.describe()  # 针对数值型的列，返回统计信息（总共多少数据、均值，标准差、最小值、最大值）
.value_counts(values,sort=True, ascending=False,normalize=False,bins=None,dropna=True) # 统计一列的不同值的个数（所以一般都是使用Series类型）。sort=True： 是否要进行排序；默认进行排序,ascending=False： 默认降序排列；,normalize=False： 是否要对计算结果进行标准化，并且显示标准化后的结果，默认是False。
#使用.value_counts后返回的值直接用关键字索引就可以得到值例如.value_counts()["男"]
# 使用.value_counts后返回的值可以使用dict转换成字典，看起来会比较舒服
df.apply(pd.value_counts) # 对每一列都进行统计，但是可读性比较差
.idmax() # 计算数据最大值、最小值所在位置的索引。一般和.value_counts结合使用可以得到


.var()      # 返回方差
series1.cov(series2)   # 计算两个序列的协方差（取值范围正负无穷）（即两个序列的趋势是否一致，趋势一致则大于0，趋势相反则小于0）
series1.corr(series2)  # 计算两个序列的相关系数（取值范围[-1,1])，等于-1（1）时完全负（正）相关，等于0时完全无关
df.cov()     # 计算df每一列与每一列之间的协方差，返回一个列数*列数的DataFrame
df.corr()    # 计算df每一列与每一列之间的相关系数（对角线上的值全部为1），返回一个列数*列数的DataFrame
```

#  十、pandas中的分组

```python
>>>res = df1.groupby(["grade","class"])   # 按照年级以及班级同时作为条件进行分组
>>>res.groups         # （接上一行）返回一个字典，key为年级与班级的联合，value为属于当前年级以及班级的行索引
>>>res.sum()   # 对每一组组内数据进行求和
>>>res.count() # 对每一组组内数据进行计数
>>>res.get_group((5,2))  # （接上一行）返回5年级2班的所有信息
>>>age_mean = res["age"].agg(np.mean) # （接上一行）按照grade和class分成的组，对每一组的age进行聚合求每一组age的平均值
>>>age_mean.loc[(3,1)]  # 获得行索引为(3,1)即3年级1班的信息，(3,1)为联合索引
```

# 十一、pandas中的表连接、表拼接

```python
"""pd.merge()  :相当于mysql中的join表连接
   left/right : 两个待合并的DataFrame对象
   how　：需要执行的合并类型
       --left ： (如果两表的同一字段下的值不完全相同，以左表为基准)    
       --right ：(如果两表的同一字段下的值不完全相同，以右表为基准)
       --outer　：(如果两表的同一字段下的值不完全相同，取两表并集)
       --inner　：(如果两表的同一字段下的值不完全相同，取两表交集)
   on : 指定列名用于连接，必须两个表都由，关联键；如果没有设置我们使用共同拥有的列名作为连接键
   left_on/right_on : 两个表中列名不一样，但是含义一样情况下的连接
"""
pd.merge(left, right, on=["name","id"],how="inner")  #将left与right进行连接，指定连接的键为name和id，表连接方式为inner(取交集)


"""concat：表拼接
    objs:被拼接的两个表，使用列表把两个表括起来
    axis:沿着哪个轴方向拼接
    join ：拼接方式（inner(索引的交集)、outer(索引的并集（默认）))
    ignore_index ：拼接后索引是否进行重排
"""
```

# 十二、数据采样

```python
.sample(n=2,axis=0,random_state=1)        #沿着0轴采样2行，设置随机数种子为1
.sample(frac=0.7,axis=0,random_state=1)   #沿着0轴采样70%，设置随机数种子为1
```

# 十三、文件读写

```python
.to_csv(path, index=False,sep=";")  # 将df数据写入文件（一般为csv),index=False代表不保留索引，sep代表存储后元素之间用分号分隔 
.to_json()
.to_excel()  # 需要pip安装pyopenxl插件
等等方法
pd.read_csv(path,index=0,sep=";",head=None，names=["A","B","C"])  #读取.txt或者.csv或者.data文件，index=0代表以文件中的第0列为索引列，head=None代表不使用第一行作为表头，sep代表使用";"作为分隔符,names代表使用["A","B","C"]作为表头
pd.read_json()  #只能读取类似于DataFrame类型的json文件
pd.read_excel()
```

