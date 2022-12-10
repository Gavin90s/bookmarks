#### docker相关
```
docker 基本操作命名
1、列出所有docker镜像文件
sudo docker images

2、拉取一个docker镜像文件
sudo docker pull pkdogcom/caffe-ssd

3、从镜像中fork一个容器，并运行这个docker容器
sudo nvidia-docker run -it --name ssd --shm-size 32G -v /home/*:/data/ pkdogcom/caffe-ssd bash
-v 表示挂载目录，上述命令将宿主机器的/home/*目录挂载到docker容器中的/data/目录
注：-v 可以多次使用，如 ***  -v /disk2/:/data/ -v /disk1/:/home/  
--shm-size 32G 避免出现 insufficient shared memory (shm) 错误
-it表示交互运行

4、列出所有可以运行的docker容器
sudo docker ps -a

5、列出所有正在运行的docker容器
sudo docker ps

6、运行一个容器
sudo docker 8f556958d818 start

7、停止一个容器
sudo docker 8f556958d818 stop

8、进入容器环境中
docker exec -it 9df70f9a0714 /bin/bash

9、退出docker环境
ctrl + P + Q
注：docker环境中，在终端执行sh命令时，命令正在处理过程中，并打印输出信息，当用ctrl + P + Q退出时，sh命令操作并不会停止，而会依然进行。这如直接在非docker环境中处理不同，在非docker环境处理时，如果终端正在执行命令，关闭终端或者断开与服务器的连接时，命令都会停止。

10、docker tag
docker tagreg.docker.alibaba-inc.com/algorithm/quake:fairseq registry.cn-hangzhou.aliyuncs.com/docker_workspace_zz/fairseq-wav2vec:fairseq

11、docker commit
sudo docker commit 4b9110329dfd registry.cn-hangzhou.aliyuncs.com/docker_workspace_zz/fairseq-wav2vec:fairseq

12、bash: vi: command not found
解决方法：
apt-get update
apt-get install vim

注：apt-get update 卡在 [0% [Working]  
解决方案: mkdir /etc/apt/sources.list.d.backup 
cp /etc/apt/sources.list.d/* /etc/apt/sources.list.d.backup
rm -rf /etc/apt/sources.list.d

docker采用volume传输数据
create a volume
    $ docker volume create my-vol
```

```
1创建完成之后可以查看详细信息
 $ docker volume inspect my-vol
 可得信息如下：注意这个Mountpoint所对应的目录就是我们用来主机和容器进行文件传输的目录。

2 然后在使用run启动一个容器的时候就可以使用该volume：
 $ docker run -it --name test -v my-vol:/hostdata nginx /bin/bash
  -v命令将刚才创建的数据卷挂载到容器中的hostdata目录下了。

3这时候我们在容器中给hostdata目录下添加文件的时候，在主机的的/var/lib/docker/volumes/my-vol/_data中就可以看到了，同理在主机的该目录中添加文件，在容器的hostdata中也可以看到。
docker文件传递
1从docker和主机之间相互传输文件
（1）先拿到容器的短ID或者指定的name.
如下图：可以看出一个docker容器为ssd，短ID为1cd9b9bd0db8
命令：sudo docker ps -a
 (2) 然后根据这两项的任意一项拿到ID全称。
命令：sudo docker inspect -f '{{.ID}}' ssd 
可得：ID的全称为 1cd9b9bd0db84a513f7d859903ab4863f58d5a2524edd74aad73fabedde9ec66
（3）有了ID的全称后，本机和容器之间的文件传输就简单了。
   1）本地文件传到docker中：
      docker cp 本地文件路径 ID全称:容器路径  
      如：sudo docker cp temp 1cd9b9bd0db84a513f7d859903ab4863f58d5a2524edd74aad73fabedde9ec66:/opt/caffe
      将本地当前目录下的temp文件传输docker ssd中的/opt/caffe目录下了。
   2）docker中文件传到本地
      docker cp ID全称:容器文件路径 本地路径
```

#### 1. 登录阿里云Docker Registry
```
$ sudo docker login --username=petty_cash registry.cn-hangzhou.aliyuncs.com
用于登录的用户名为阿里云账号全名，密码为开通服务时设置的密码。
您可以在访问凭证页面修改凭证密码。
```
#### 2. 从Registry中拉取镜像
```
$ sudo docker pull registry.cn-hangzhou.aliyuncs.com/docker_workspace_zz/fairseq-wav2vec:[镜像版本号]
```

#### 3. 将镜像推送到Registry
```
$ sudo docker login --username=petty_cash registry.cn-hangzhou.aliyuncs.com
$ sudo docker tag [ImageId] registry.cn-hangzhou.aliyuncs.com/docker_workspace_zz/fairseq-wav2vec:[镜像版本号]
$ sudo docker push registry.cn-hangzhou.aliyuncs.com/docker_workspace_zz/fairseq-wav2vec:[镜像版本号]
请根据实际镜像信息替换示例中的[ImageId]和[镜像版本号]参数。
```

#### 4. 选择合适的镜像仓库地址
```
从ECS推送镜像时，可以选择使用镜像仓库内网地址。推送速度将得到提升并且将不会损耗您的公网流量。
如果您使用的机器位于VPC网络，请使用 registry-vpc.cn-hangzhou.aliyuncs.com 作为Registry的域名登录。
```

#### 5. 示例
使用"docker tag"命令重命名镜像，并将它通过专有网络地址推送至Registry。
```
$ sudo docker tag 37bb9c63c8b2 registry-vpc.cn-hangzhou.aliyuncs.com/acs/agent:0.7-dfb6816
```

使用 "docker push" 命令将该镜像推送至远程。
```
$ sudo docker push registry-vpc.cn-hangzhou.aliyuncs.com/acs/agent:0.7-dfb6816
```

```
sudo docker run -it --name wenet_test --shm-size 32G -v /home/zhuozhu.zz:/workspace/zhuozhu.zz reg.docker.alibaba-inc.com/algorithm/quake:torch-1.8.2-cuda111-centos7-train-v2.0 bash
sudo docker exec -it f938098307dd /bin/bash
sudo docker commit asr reg.docker.alibaba-inc.com/docker_workspace_zz/espnet-ctc-att-cuda11-quake:v0.1
sudo docker push reg.docker.alibaba-inc.com/docker_workspace_zz/espnet-ctc-att-cuda11-quake:v0.1
```
