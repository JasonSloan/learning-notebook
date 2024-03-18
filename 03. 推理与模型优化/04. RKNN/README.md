# 一. 参考文档

**RKNN操作步骤参考:**

**[官方github](https://github.com/rockchip-linux/rknn-toolkit2)提供的RKNPU2_SDK中的v1.6.0(或者更高的release版本)下的Quick_Start文档**

# 二. ssh的安装

```bash
 apt-get remove openssh*
 apt install openssh-client
 apt install openssh-server
```

# 二. 模型编译

**编译环境搭建: **

创建一个新的conda环境, 指定好python版本, 按照本文件夹下的requirements对应的python版本安装, 建议指定使用阿里源, 否则安装会出现 No matching distribution found for tf-estimator-nightly==2.8.0.dev2021122109
**模型编译:**

使用该文件夹下的convert.py



