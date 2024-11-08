## 1. view()与reshape()的区别理解

两者都是用来重塑tensor的shape的。view只适合对满足连续性条件（contiguous）的tensor进行操作，而reshape同时还可以对不满足连续性条件的tensor进行操作，具有更好的鲁棒性。view能干的reshape都能干，如果view不能干就可以用reshape来处理。但view的好处是不会对原始tensor的数据拷贝到新的地址, 只是对原始tensor数据进行一个新的视图引用; reshape会直接深拷贝原始tensor的数据

[参考链接](https://blog.csdn.net/Flag_ing/article/details/109129752)

### 2. permute()与transpose()的区别理解

区别1: transpose()只能一次操作两个维度；permute()可以一次操作多维数据

```python
# 对于transpose
x.transpose(0,1)     'shape→[3,2] '  
x.transpose(1,0)     'shape→[3,2] '  
y.transpose(0,1)     'shape→[3,2,4]' 
y.transpose(0,2,1)  'error，操作不了多维'

# 对于permute()
x.permute(0,1)     'shape→[2,3]'
x.permute(1,0)     'shape→[3,2], 注意返回的shape不同于x.transpose(1,0) '
y.permute(0,1)     "error 没有传入所有维度数"
y.permute(1,0,2)  'shape→[3,2,4]'
```

区别2: transpose()中的dim代表的是我要交换哪两个维度；permute()中的dim代表的是交换后新的维度应该是什么顺序

```python
# 对于transpose，不区分dim大小
x1 = x.transpose(0,1)   'shape→[3,2] '  
x2 = x.transpose(1,0)   'shape→[3,2] '  
print(torch.equal(x1,x2))
' True ，value和shape都一样'

# 对于permute()
x1 = x.permute(0,1)     '不同transpose，shape→[2,3] '  
x2 = x.permute(1,0)     'shape→[3,2] '  
print(torch.equal(x1,x2))
'False，和transpose不同'

y1 = y.permute(0,1,2)     '保持不变，shape→[2,3,4] '  
y2 = y.permute(1,0,2)     'shape→[3,2,4] '  
y3 = y.permute(1,2,0)     'shape→[3,4,2] '  
```



