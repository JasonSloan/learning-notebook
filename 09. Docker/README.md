**参考文档，写的很好：<https://docker.easydoc.net/doc/81170005/cCewZWoN/lTKfePfP>**

# 一. 安装

```python
windows上安装：https://docker.easydoc.net/doc/81170005/cCewZWoN/lTKfePfP
Linux上安装：CSDN--->收藏-->Docker(注意，先按照第一篇《Ubuntu系统安装docker》进行普通docker的安装，然后再按照第二篇《nvidia-docker配置深度学习环境服务器》中的章节"2.3 安装nvidia-docker2"安装nvidia-docker，第二篇只需要执行该章节就行，当run一个docker的时候需要指定参数--gpus all，这样容器中就可以使用显卡了)
docker命令详解：https://docs.docker.com/engine/reference/commandline/run/

命令行登陆docker:
docker login -u <username>  # username是docker hub上的昵称
启动docker服务ls
sudo service docker start
```

# 二、例子

## 1. docker拉取python镜像到本地并运行，安装第三方库

```bash
# 拉取镜像到本地
docker pull python:3.7
# 运行镜像，privileged代表给与容器一些权利（这样就可以在容器中挂载），it代表打开交互模式，d代表后台运行，-v代表挂载，具体为将宿主机的/opt/python_home挂载到镜像容器中的/home/python_home中，挂载后在宿主机/opt/python_home下产生的文件也相当于在容器中的/home/python_home下产生
docker run --privileged -itd --name python3  -v /opt/python_home:/home/python_home python:3.7
#　执行镜像，进入到镜像容器内部
docker exec -it python3 /bin/bash
# 在镜像中安装第三方库
pip install 
# 在镜像中进入python
python
# 退出容器镜像
exit
```

## 2. Dockerfile

```bash
官方文档：https://docs.docker.com/engine/reference/builder/
CSDN文档：https://blog.csdn.net/m0_46090675/article/details/121846718?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168406712116800197089082%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168406712116800197089082&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-121846718-null-null.142^v87^koosearch_v1,239^v2^insert_chatgpt&utm_term=Dockerfile&spm=1018.2226.3001.4187
```

## 3. 打包发布自己的镜像流程

```bash
先自行打一个底，比如使用python3.7镜像作为底，那么就拉取这个镜像，运行镜像，进入镜像，然后手动一步一步安装所需的库，如果没问题了，那么就把它写成Dockerfile文件，然后build成镜像，再push到远程仓库

步骤：
如果出现permission denied 就使用sudo
命令行登陆docker:
sudo docker login -u <username>        (我的docker hub的username是jasonsloan)
启动docker服务
sudo service docker start
拉取基础镜像
docker pull python:3.7
查看已有镜像
sudo docker images
运行基础镜像，并将容器中/home/python_home与本地./映射
// 当需要使用gpu的时候，需要指定参数--gpus all ：sudo docker run -itd --gpus all --name test jasonsloan/gpu_env
docker run --privileged -itd --name face_detection -v ./:/home/python_home python:3.7  
// 也可以直接run不做目录映射
// docker run --privileged -itd --name face_detection python:3.7
启动已运行的镜像
sudo docker start face_detection
查看正在运行的镜像ID(所有的名字都可以替换成ID来运行)
sudo docker ps
// 停止docker
// sudo docker stop face_detection
// 删除docker镜像
// sudo docker rm face_detection
进入基础镜像
docker exec -it face_detection /bin/bash
进入基础镜像中与宿主机./映射的文件夹/home/python_home
cd home/python_home/
拷贝本地文件到镜像中
① docker ps         先查看镜像ID
② sudo docker cp ./Anaconda3-2023.07-2-Linux-x86_64.sh 8b202fb3d59e:/home     再拷贝文件
各种pip安装
pip install ...
如果在该容器中运行项目没问题，那么就开始按照步骤制作Dockerfile
在网页上登录docker hub，创建一个仓库名字为face_detection
Dockerfile制作完成后开始build编译，将当前文件夹下所有文件编译成名为face_detection的镜像
docker build -t face_detection:v1 .
编译结束运行看是否有错误，8080是本地端口，22是容器端口
docker run --privileged --shm-size=4g -it --name face --gpus all -p 8080:22 face_detection:v1  // 当需要使用gpu的时候，需要指定参数--gpus all; 指定shm-size,在容器中使用df -h可以看到shm的值为8G,否则默认为64M,太小不够用
启动容器
sudo docker start face
查看CONTAINER ID
docker ps
将镜像打成标签：其中{image_id}为build好的镜像的哈希值，jasonsloan为docker hub上的用户名，face_detection:v1为docker hub上的仓库名
docker tag {image_id} jasonsloan/face_detection:v1
push到远程仓库，push前必须tag，所以push的时候就不用指定是哪个image了，只需要指定是远程哪个仓库
docker push jasonsloan/face_detection:v1
```

## 4. 一个可以打包Ubuntu、trtpy、cuda、tensorrt、anaconda、pytorch、OpenCV环境,配置git代理的Dockerfile

```bash
使用方法：
服务器拉取：
sudo docker login
sudo docker pull jasonsloan/gpu_env			（私有仓库）
sudo docker run --privileged --gpus all -itd --name yolo_env -p 2345:22 jasonsloan/gpu_env
sudo docker ps
本地连接：
ssh -p 2345 root@服务器IP地址
```

Dockerfile：（**注意下面的trtpy set-key testkey中的testkey是假的，实际的key不公开，当需要build镜像的时候，注意替换为实际的key**）

```bash
FROM ubuntu:22.04

# 安装一些包
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    ssh \
    cmake \
    vim \
    openssh-server \
    g++ \
    zip \
    unzip 


# 安装 OpenCV
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libopencv-dev

# 安装 Miniconda
ENV CONDA_DIR=/root/software/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh -O /root/miniconda3.sh
RUN /bin/bash /root/miniconda3.sh -b -p $CONDA_DIR && \
    rm -rf /root/miniconda3.sh && \
    export PATH="/root/software/miniconda3/bin:$PATH"

# 设置环境变量
ENV PATH="/root/software/miniconda3/bin:$PATH"

# 配置 Conda 镜像源
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --set show_channel_urls yes

# 创建trtpy环境,安装trtpy
RUN /root/software/miniconda3/bin/conda run -n base pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    /root/software/miniconda3/bin/conda run -n base pip install trtpy -U 

# 设置trtpy密钥,使用trtpy配置CUDA,TensorRT环境
RUN /root/software/miniconda3/bin/conda run -n base python -m trtpy set-key testkey && \
    /root/software/miniconda3/bin/conda run -n base python -m trtpy get-env

# 安装 PyTorch, TorchVision, OpenCV
RUN /root/software/miniconda3/bin/conda run -n base pip install torch torchvision opencv-python

# 初始化 Conda
RUN conda init bash

# 设置root密码
RUN echo 'root:111111' | chpasswd

# 配置ssh,使其可以使用root用户登录
RUN mkdir /var/run/sshd && \
    echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# 配置环境变量,使其每次开启容器时自动启动ssh服务
RUN echo "/etc/init.d/ssh start" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc" 

# 安装fastgithub
RUN wget https://github.com/JasonSloan/learning-notebook/releases/download/v1/fastgithub_linux-x64.zip -O /root/software/fastgithub_linux.zip && \
    cd /root/software && \
    unzip fastgithub_linux.zip && \
    rm -rf fastgithub_linux.zip 

# 配置环境变量,使其每次开启容器时自动启动fastgithub
RUN echo "cd /root/software/fastgithub_linux-x64 && nohup ./fastgithub >/dev/null 2>log &" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc" 

# 为git添加代理
RUN git config --global http.proxy http://127.0.0.1:38457

# 简化python -m trtpy启动命令为trtpy
RUN echo alias trtpy=\"python -m trtpy\" >> ~/.bashrc 

# CMD
CMD ["/bin/bash"]
```

## 5. 给已运行的 docker 容器修改或增加端口映射

```bash
sudo docker stop {容器的名称或者 id }
# 如果cd不进去，那就chmod 777 folder_name
cd /var/lib/docker/containers/{hash_of_the_container}
chmod 777 hostconfig.json
vim hostconfig.json
# 在hostconfig.json 配置文件中，找到 "PortBindings":{} 这个配置项，然后进行修改。我这里添加了两个端口映射，分别将宿主机的 8502 端口以及 8505 端口映射到容器的 8502 端口和 8505 端口。HostPort 对应的端口代表宿主机的端口。
{
    "PortBindings": {
        "8502/tcp": [
            {
                "HostIp": "",
                "HostPort": "8502"
            }
        ],
        "8505/tcp": [
            {
                "HostIp": "",
                "HostPort": "8505"
            }
        ]
    }
}
# 如果有config.v2.json 配置文件或者 config.json 配置文件中也记录了端口，也需要进行修改，如果没有，就不需要改。
# 只需要修改 "ExposedPorts": {} 相关之处。
{
    "Args": [],
    "Config": {
        "ExposedPorts": {
            "8502/tcp": {},
            "8505/tcp": {}
        },
        "Entrypoint": [
            "/bin/sh"
        ]
    }
}

# 最后重启 docker
sudo service docker restart
# 再次启动容器
sudo docker start {hash_of_the_container}
```

## 6. 查询已运行容器的内在的IP

```bash
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' containerID
```

