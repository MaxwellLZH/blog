一天一元搭建你自己的深度学习服务

打开阿里云服务器ECS网站(https://www.aliyun.com/product/ecs)


系统配置：

创建密钥完成后，下载私钥并保存好！保存好！保存好！（重要的事情说三遍）


分组设置：
暂时先不做改动


付款页面：
保存为模版





进入管理控制台（https://ecs.console.aliyun.com/#/home) 在**我的资源**里可以看到我们刚创建的实例。





登录服务器：


进入根目录下的.ssh目录，如果当前目录下没有一个叫config的文件的话需要新创建一个：

cd ~/.ssh
touch config   # 创建config文件

在config文件中加上我们的实例信息:

Host dls
HostName 139.196.40.202
Port 22
User root
IdentityFile /Users/Max/Documents/Projects/blog/deep-learning-api/deeplearningserver.pem


添加完成后使用ssh dls就可以登录我们的实例啦!【登录图片】如果需要退出的话，打exit就好了:)


Windows系统怎么使用密钥请参考（https://help.aliyun.com/document_detail/51798.html?spm=a2c4g.11186623.2.12.7bc1388dSUrc0f#concept-ucj-wrx-wdb）


怎么将本地代码传到服务器上？
scp -F ~/.ssh/config -r ./dls dls:/root


tree ./dls

如果提示错误找不到`tree`命令的话可以用一下方法安装：
sudo apt update   # 更新repo信息
sudo apt install tree




配置我们的服务器：


1. 安装anaconda

打开anaconda的网站（https://www.anaconda.com/distribution/）选择**Linux**，之后右键Python3.7版本下的**64-Bit(x86)Installer**，复制链接地址

随后在terminal中使用curl -O [链接]来下载文件，此处-O是使本地文件名和anaonda服务器上的文件名保持一致。



```
# 安装Anaconda
# 链接可在https://www.anaconda.com/distribution上找到最新的链接地址
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh

# 安装Anaconda
bash Anaconda3-2019.10-Linux-x86_64.sh

# Anaconda安装会修改.bashrc文件，重新执行一下
source ~/.bashrc
```


2. 安装项目需要的包
python -m pip install -r dls/requirements.txt

python -c "import tensorflow; print(tensorflow.__version__)"




[Optional] 怎么在Chrome上给你的服务器起一个昵称
在Chrome强大的插件市场中有一个URL Alias能够做到这一点，在chrome web store(https://chrome.google.com/webstore/category/extensions)中搜索并安装URL Alias, 











让程序在后台一直运行：
nohup uvicorn main:app --host 0.0.0.0 --port 80 &

重新登录之后怎么关停：
ps -ef | grep uvicorn 找univorn对应的PID
kill {PID}










fastapi介绍
It simply works，一个简单的hello world例子


三种不同的parameter: Path, Query和Body






ssh -i ~/.ssh/dls.pem root@139.196.40.202