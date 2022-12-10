# note

#### mac 查询本机地址 
ifconfig

#### 最小化窗口显示
command + M 返回桌面，最小化窗口显示。

#### tensorboard 查看日志
```
tensorboard --logdir="../mdl/result/albert_small_zh_bs32_lr5e-05_epoch10/"
```

#### python格式化工具
```
pip install yapf
yapf --style='{based_on_style: google, indent_width: 2}' optimization_google.py > optimization_google_.py
yapf --style='{based_on_style: pep8, indent_width: 2}' optimization_google.py > optimization_google_.py
```

#### conda 使用
```
conda list -e > requirements.txt
conda create --name <env> --file requirements.txt
conda activate <env>
conda install pip
pip freeze > requirements.txt
python setup.py install
```

#### vim 高亮搜索结果
nohlsearch 或者：set nohlsearch  

#### conda 添加清华源
````conda添加清华源
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
  conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
  conda config --set show_channel_urls yes
````

#### awk 使用
```
cat xxx.log | awk -F' ' '{for (i=1;i<=NF;i++){print $i}}' 
```

#### srilm 训练
srilm 训练 https://zhuanlan.zhihu.com/p/136734757

$cat train_big_lm.sh
```
!/bin/bash
split -l 2000000 zh_hans.splited.txt ngramlm_traindir/
find . > ngram.train.data.list
SRILM/bin/make-batch-counts ngram.train.data.list 20 cat counts -order 3 -sort
SRILM/bin/make-big-lm -read ./ngram_counts/*.ngrams.gz -lm ./big.split.lm -order 3  -interpolate -kndiscount -write big.split.count
gzip -c big.split.lm > big.split.lm.gz
SRILM/lm/bin/i686-m64/ngram -lm jiankang.0.6.lm.gz -order 3 -ppl test.txt -debug 2
```

#### sql 设置 lifecycle
```
alter table <table_name> set lifecycle <days>;
```

#### mozilla 开源语音数据集
```
https://commonvoice.mozilla.org/zh-CN/datasets
```

#### 阿里云安全辱骂检测挑战赛
https://tianchi.aliyun.com/competition/entrance/231762/information?spm=a2c22.13151069.1383125.3.54d85f74nCDZcf


#### 获取路径
```
cat wav.scp | awk -F'\t' '{print $2}' |  sed 's#\(.*\)/.*#\1#' | sort | uniq
```

#### coredump 文件
ulimit -c unlimited

cmake -DCMAKE_INSTALL_PREFIX=/disk4/zhuozhu.zz/tops/sentencepiece ..

#### docker虚拟环境
```
pip3 install soundfile
pip3 install editdistance
yum install libsndfile
cffi==1.14.4
pip3 install Cython==0.29.23
pip3 install setuptools==56.0.0
pip3 install wheel==0.36.2
pip3 install omegaconf==2.0.6
pip3 install importlib-resources==5.1.2
pip3 install antlr4-python3-runtime==4.8
pip3 install hydra-core==1.0.6
pip3 install sacrebleu==1.5.1
```
importlib-resources, omegaconf, hydra-core, sacreble

#### git 使用
```
git 单个文件版本控制
#git 还原某个文件
git checkout -- freesound_download.py

#首先查看文件的历史版本
git log /path/to/file

#还原到某个版本
git checkout ${commit} /path/to/file
git 远程仓库管理
# 查看本地添加了哪些远程
git remote -v
origin	http://gitlab.alibaba-inc.com/zhuozhu.zz/wav2vec_libri.git (fetch)
origin	http://gitlab.alibaba-inc.com/zhuozhu.zz/wav2vec_libri.git (push)

# 添加远程仓库
git remote add all git@gitlab.alibaba-inc.com:zhuozhu.zz/fairseq.git

$git remote -v
all	http://gitlab.alibaba-inc.com/zhuozhu.zz/fairseq.git (fetch)
all	http://gitlab.alibaba-inc.com/zhuozhu.zz/fairseq.git (push)
origin	https://github.com/pytorch/fairseq.git (fetch)
origin	https://github.com/pytorch/fairseq.git (push)

# 删除远程仓库
git remote remove all

# 推送到远程仓库
git push all

# 删除远程分支
git push all --delete espnet_ctc_att_ssl_v1.02

# 删除本地分支
git branch -D espnet_ctc_att_ssl_v1.02

# 比较两个不同branch的代码
git diff espnet_ctc_att_ssl_roberta espnet_ctc_att_ssl_pretrained_decoder espnet2/asr/decoder/transformer_decoder.py

# 比较两个不同branch的文件差异
git diff espnet_ctc_att_ssl_roberta espnet_ctc_att_ssl_pretrained_decoder --stat

# 拉取某个branch的文件
git checkout espnet_ctc_att_ssl espnet2/asr/decoder/transformer_decoder.py

# branch 重命名
git branch -m branch_name

# 删除远程分支
git push --delete origin oldbranch
```

#### cmu词典资源
https://cmusphinx.github.io/wiki/tutorialdict/#using-g2p-seq2seq-to-extend-the-dictionary
https://github.com/cmusphinx/cmudict/blob/master/cmudict.phones

#### vim 使用
```
vim替换命令
https://zhuanlan.zhihu.com/p/61515833
#整个文件范围内替换
:%s /search/replace/g

#在第5行~15行间进行替换
:5,15s/dog/cat/g

#当前行~文件结尾进行替换
:.,$s/dog/cat/g

#在后续9行进行替换
:.,+9s/dog/cat/g

#将空格替换成逗号
:%s/\s/,/g

#将多个连续空格替换成逗号
:%s/\s+/,/g

#将换行符替换成逗号
:%s/\r/,/g

#1为起始行号，315为终止行号，
# ^在正则中代表行首，
# \s*代表若干个空格，可以没有，
# [0-9]*代表若干个数字，可以没有，
# 即将^\s*[0-9]*\s*替换为NOTHING。
%1,315s/^\s*[0-9]*\s*//g

vim 选择括号内的内容
https://blog.csdn.net/weixin_39675038/article/details/111526130
set mouse=a，prevents the ability of copying and pasting out of vim with readable characters.
set mouse=r, should fix your issue with that.
```

#### shell 添加注释
```
#!/bin/bash
echo "Say Something"
<<COMMENT
    your comment 1
    comment 2
    blah
COMMENT
echo "Do something else"
```

TorchElastic - 弹性、容错的分布式训练
https://zhuanlan.zhihu.com/p/156060169?utm_source=wechat_session

Python调试器pdb
https://zhuanlan.zhihu.com/p/37294138

#### 将flac转化成wav
```
$cat flac_to_wav.sh
# Convert all .flac files within this folder to .wav files
find ./test-clean -iname "*.flac" | wc
for flacfile in `find ./test-clean -iname "*.flac"`
do
#ffmpeg-i $flacfile -ab 64k -ac 1 -ar 16000 -f wav "${flacfile%.*}.wav"
echo "convert $flacfile to wav."
sox $flacfile -r 16000 -b 16 -c 1 "${flacfile%.*}.wav"
done
```

#### 结巴分词
https://blog.csdn.net/meiqi0538/article/details/80213431

#### 批量杀进程
```
ps -ef | grep aaa | grep -v grep | awk '{print "kill -9 " $2}' | sh
```

#### python读取键盘输入
```
y=input("请输入y=")
```

#### python hash_map 一键多值
```
from collections import defaultdict
dict_one_to_more = defaultdict(list)
for x in range(10)
	dict_one_to_more["Key"].append("Value：" + str(x))
 ```
 
#### grep 取文件的交集、差集
```
# grep命令是常用来搜索文本内容的，根据输入的pattern，输出命中的内容。
#求两个文件的交集。
$ grep -F -f a.file b.file
c
d
e

#差集B-A
$ grep -F -v -f a.file b.file
f
g

#差集A-B
$ grep -F -v -f b.file a.file
a
b
```

#### glob 使用
```
import glob
#父目录中的.py文件
f = glob.iglob(r'../*.py')
print (f) # <generator object iglob at 0x00B9FF80>
for py in f:
    print (py)
```

#### python 参数中*和**的作用说明 
```
https://blog.csdn.net/yilovexing/article/details/80577510
*args 与 **kwargs 的区别，两者都是 python 中的可变参数：
*args 表示任何多个无名参数，它本质是一个 tuple
**kwargs 表示关键字参数，它本质上是一个 dict
>>> def fun(*args, **kwargs):
...     print('args=', args)
...     print('kwargs=', kwargs)
... 

>>> fun(1, 2, 3, 4, A='a', B='b', C='c', D='d')
args= (1, 2, 3, 4)
kwargs= {'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd'}
```

#### 将MP4转wav
```
./ffmpeg -i Elton_Britt_The_Red_We_Want_is_the_Red_We_Got.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 Elton_Britt_The_Red_We_Want_is_the_Red_We_Got.wav
convert_wav_format.py 

# coding=utf-8
import os
import glob

#将webm转化wav 
f = glob.iglob(u'/Users/zhuozhu/Downloads/音频违规样本/愿荣光归香港/*.webm')
for s in f:
  ss=s.replace("webm", "wav")
  os.system(u"./ffmpeg -i " + s + " -c:a pcm_s16le -ar 16000 -ac 1 " + ss)

#将m4a转化wav 
f = glob.iglob(u'/Users/zhuozhu/Downloads/音频违规样本/愿荣光归香港/*.m4a')
for s in f:
  ss=s.replace("m4a", "wav")
  os.system(u"./ffmpeg -i " + s + " -vn -acodec pcm_s16le -ar 16000 -ac 1 " + ss)
    
#将webm转化wav 
f = glob.iglob(u'/Users/zhuozhu.zz/Downloads/prohabit_music/*.mp*')
for s in f:
  print(s)
  ss=s.replace("mp3", "wav").replace("mp4", "wav")
  os.system(u"ffmpeg -i " + s + " -acodec pcm_s16le -ar 16000 -ac 1 " + ss)
```

#### 删除文件名中空格
```
for oldname in *
do
  newname=`echo $oldname | sed -e 's/ /_/g'`
  mv "$oldname" "$newname"
done
```

离线物理机
ssh zhuozhu.zz@11.238.144.94
代码调试工具torchsnooper

python fire 库
import fire

def hello(name):
  return 'Hello {name}!'.format(name=name)

def main():
  fire.Fire(hello)
不再需要指定hello函数，因为我们调用了fire.Fire(hello)
espnet ASR模型训练目录
b05b01460.nt12：/data7/zy134358/work/espnet/egs2/ai shell/asr/data/train

oracle WER 
计算asr decoding lattice 和hypothesis之间的关系。
it prints out the corresponding WERs in its logging output. This is done by constructing an "edit-distance FST" and composing this with unweighted acceptors constructed from the lattice and from the reference.
danielpovey主页
http://www.danielpovey.com/publications.html
huggingface tranformers安装
import error “version `GLIBC_2.29’ not found” 解决方法
conda install -c conda-forge transformers
conda install importlib-metadata
huggingface tranformers将 tf 模型转化为 pytorch模型
https://huggingface.co/transformers/converting_tensorflow_models.html
transformers-cli convert --model_type bert \
                         --tf_checkpoint roberta_tiny_50G_whole_model.ckpt.index \
                         --config bert_config_tiny.json \
                         --pytorch_dump_output pytorch_model.bin
roberta 代码
https://github.com/Gavin90s/CLUEPretrainedModels/blob/b384fd41665a8261f9c689c940cf750b3bc21fce/baselines/models/roberta/modeling.py#L754
bart 代码
https://github.com/huggingface/transformers/blob/master/src/transformers/models/bart/modeling_bart.py
自监督预训练（三）wav2vec 2.0原理剖析
https://blog.csdn.net/xmdxcsj/article/details/115787729
CLUECorpus2020
A Large-scale Chinese Corpus for Pre-training Language Model
集团docker地址
https://docker.alibaba-inc.com/#/dockerImage/2955998/detail

Scheduled Sampling
主要应用在序列到序列模型的训练阶段，而生成阶段则不需要使用。训练阶段解码器在最大化第t个元素概率时，标准序列到序列模型使用上一时刻的真实元素yt−1作为输入。设上一时刻生成的元素为gt−1，Scheduled Sampling算法会以一定概率使用gt−1作为解码器输入。
https://blog.csdn.net/zlrai5895/article/details/84748749
查看压缩包内容
tar -tf xxxx.tar.gz
prefix beam search
https://blog.csdn.net/weixin_42615068/article/details/93767781
解决sort磁盘空间不足
export TMPDIR=/tmp
pytorch 调试
import torchsnooper
@torchsnooper.snoop()
调研报告｜在线语音识别改进之 RNN-T 训练
https://zhuanlan.zhihu.com/p/146832796




多线程压缩
#下载pigz
wget https://download-ib01.fedoraproject.org/pub/epel/7/x86_64/Packages/p/pigz-2.3.4-1.el7.x86_64.rpm
    
#安装pigz    
rpm -ivh pigz-2.3.4-1.el7.x86_64.rpm 

#多线程压缩
tar cf - wenetspeech | pigz -9 -p 32 > /data6/zhuozhu.zz/wenetspeech_8k.tar.gz

#多线程解压
pigz -dc wenetspeech_8k.tar.gz | tar xf -
gdb 调试
gdb test.bin pid.core
VScode c++环境
wget https://vscode.cdn.azure.cn/stable/f4af3cbf5a99787542e2a30fe1fd37cd644cc31f/VSCode-darwin-universal.zip

#使用clang-format插件
    
#安装 clang-format, 用code formatter
wget http://releases.llvm.org/8.0.0/clang+llvm-8.0.0-x86_64-apple-darwin.tar.xz
tar xvfJ clang+llvm-8.0.0-x86_64-apple-darwin.tar.xz -C ./clang-format
ln -s /Users/zhuozhu/clang-format/clang+llvm-8.0.0-x86_64-apple-darwin/bin/clang-format /usr/local/bin/clang-format
opustool安装
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
wget https://codeload.github.com/xiph/opus-tools/zip/refs/heads/master
wget https://ftp.osuosl.org/pub/xiph/releases/ogg/libogg-1.3.5.tar.gz

git clone https://github.com/xiph/opusfile.git
git clone https://github.com/xiph/libopusenc.git
git clone https://github.com/xiph/flac.git

opusenc --bitrate 6 --max-delay 10 hkust/dev/data/format.1/20040503_222707_A000687_B000688-B-000391-001021.wav 20040503_222707_A000687_B000688-B-000391-001021.opus
查看python包路径
import a_module 
print(a_module.__file__)
在模型中将某些参数设置为不可学习但保存在模型中
self.register_buffer('my_buffer', buffer) 

https://huggingface.co/microsoft/wavlm-base-plus-sv
https://arxiv.org/pdf/2103.07552.pdf
https://github.com/Gavin90s/t5-pegasus-chinese
https://www.jiqizhixin.com/articles/2020-11-17-3
https://huggingface.co/docs/transformers/model_doc/t5
https://github.com/Gavin90s/Chinese-Advertising-Text-MultiClass-Classification/blob/master/data_processing/translate_tencent.py
https://arxiv.org/pdf/2104.06313.pdf
https://github.com/chathudan/GoogleTranslate
https://github.com/Gavin90s/mulda
https://arxiv.org/pdf/2012.05958.pdf
https://thegradient.pub/prompting/
https://arxiv.org/abs/2001.07676
https://zhuanlan.zhihu.com/p/366771566
$sudo nvidia-docker run -it --name espnet_env --network=host --shm-size 32G -v /data2/zhuozhu.zz/quake_wav2vec:/workspace/train reg.docker.alibaba-inc.com/algorithm/quake:torch-1.8.0-cuda102-cudnn7-centos7-train-v2.0 bash

linux 终端使用技巧
https://blog.csdn.net/qq_26399665/article/details/81063211
Ctrl – a ：移到行首
Ctrl – e ：移到行尾
Ctrl – k ：由光标所在位置开始，删除右方所有的字符，直到该行结束。
Ctrl – u ：由光标所在位置开始，删除左方所有的字符，直到该行开始。
Ctrl -a + Ctrl -k 或 Ctrl -e + Ctrl -u 或 Ctrl -k + Ctrl -u 组合可删除整行。
Ctrl-L：进行清屏操作
Ctrl-d 由光标位置开始，往右删除单词。往行尾删
Ctrl – y ：粘贴之前删除的内容到光标后。
粘贴 Shift-Command-V
Using page-locked host memory
In every example until this point, we have used the malloc function to allocate memory on the host, which allocates standard pageable memory on the host. CUDA provides another API called cudaHostAlloc(), which allocates page-locked host memory or what is sometimes referred to as pinned memory. It guarantees that the operating system will never page this memory out of this disk and that it will remain in physical memory. So, any application can access the physical address of the buffer. This property helps the GPU copy data to and from the host via Direct Memory Access (DMA) without CPU intervention. This helps improve the performance of memory transfer operations. 
查看.so、.a 文件的函数
nm -C --defined-only ../extra/kaldi/src/base/kaldi-base.a | grep "RandInt"

ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=nokey=1:noprint_wrappers=1 10027992976_10027992977-03-26-21.mkv
ffprobe -v error -select_streams a:1 -show_entries stream=codec_name -of default=nokey=1:noprint_wrappers=1 10027992976_10027992977-03-26-21.mkv

ffmpeg -i 10027992958_10027992959-03-26-14.mkv -map 0:a:0 xxx.wav
ffmpeg -i 10027992958_10027992959-03-26-14.mkv -map 0:a:1 yyy.wav

pip 清华源下载
pip install sklearn -i https://pypi.tuna.tsinghua.edu.cn/simple 

THOP: 统计 PyTorch 模型的 FLOPs 和参数量
https://blog.csdn.net/yiran103/article/details/97756720

python map 用法
def f(x):
  return x*x

a = map(f, [1, 2, 3])
for i in a:
  print(i)

输出结果：
1
4
9

注意：map()函数不改变原有的 list，而是返回一个新的 list。
利用map()函数，可以把一个 list 转换为另一个 list，只需要传入转换函数。


在catch中打印完整堆栈：
catch (Exception e) {
StringWriter errorsWriter = new StringWriter();
e.printStackTrace(new PrintWriter(errorsWriter));
Logger.info("get exception in matchTopicDaEQuan: [%s]", errorsWriter.getBuffer().toString());
return RetCode.RE_ERROR;
}


conda env list
conda env list
source activate pytorch
conda list
 
conda config --show
pip show pip 
pip install -r requirements.txt -t /home/zhuozhu.zz/anaconda2/envs/pytorch/lib/python2.7/site-packages
 
anaconda search -t conda tensorflow
anaconda show  cjj3779/tensorflow-gpu
 
conda install --channel https://conda.anaconda.org/cjj3779 tensorflow-gpu
conda create -n tf1.4py2.7 python=2.7
conda remove -n tf1.4py2.7 --all
conda install -n tf1.4py2.7 --channel https://conda.anaconda.org/cjj3779 tensorflow-gpu
 
conda install --channel https://conda.anaconda.org/nvidia nccl=2.4.6.1
 
pip install tensorflow-gpu==1.4.0
 
 
pytorch                  /home/zhuozhu.zz/anaconda2/envs/pytorch
tf1.0py2.7               /home/zhuozhu.zz/anaconda2/envs/tf1.0py2.7
tf1.11py2.7              /home/zhuozhu.zz/anaconda2/envs/tf1.11py2.7
tf1.4py2.7               /home/zhuozhu.zz/anaconda2/envs/tf1.4py2.7
tf1.4py3.6            *  /home/zhuozhu.zz/anaconda2/envs/tf1.4py3.6
 
 
$ pip install tensorflow # Python 2.7; CPU support (no GPU support)
$ pip3 install tensorflow # Python 3.n; CPU support (no GPU support)
$ pip install tensorflow-gpu # Python 2.7; GPU support
$ pip3 install tensorflow-gpu # Python 3.n; GPU support
 
conda install -c anaconda cudatoolkit=8
conda install -c anaconda cudnn=6
 
$pip list
DEPRECATION: The default format will switch to columns in the future. You can use --format=(legacy|columns) (or define a format=(legacy|columns) in your pip.conf under the [list] section) to disable this warning.
singledispatch (3.4.0.3)
six (1.12.0)
tensorflow-gpu (1.3.0)
tensorflow-tensorboard (0.1.8)
 
pip show tensorflow-gpu
 
conda 打包虚拟环境

Plain Text
复制代码
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
# 把虚拟环境 my_env 打包为 
my_env.tar.gz conda pack -n my_env 
# -o 参数指定打包路径和名称，把虚拟环境 my_env 打包为 out_name.tar.gz 
conda pack -n my_env -o out_name.tar.gz 
# 把某个特定路径的虚拟环境打包为 my_env.tar.gz 
conda pack -p /explicit/path/to/my_env
 
# 创建目录 `my_env`，并将环境解压至该目录 
mkdir -p my_env 
tar -xzf my_env.tar.gz -C my_env 
# 使用python而不激活或修复前缀。 
# 大多数 python 库可以正常工作，但需要处理前缀的部分将失败。 
./my_env/bin/python 
# 激活环境，同时这步操作会将路径 `my_env/bin` 添加到环境变量 path 
source my_env/bin/activate 


crawler
#!/bin/python


coding: utf-8

import os
import re
import sys
import urllib2

import socket
socket.setdefaulttimeout(10.0)

MAXNUM = 10

def download(index):
url = 'http://dongman.2345.com/lt/%d' % (index)
try:
html = urllib2.urlopen(url).read()
except Exception:
html = "None"
return html

def parse(html):
videos = []
html = html.strip()
pattern = r'<a title="(.+?)"' find_re = re.compile(pattern, re.DOTALL) for item in find_re.findall(html): result = dict( title = item.decode('gbk') ) videos.append(result) return videos

if name == 'main':
for i in range(MAXNUM):
html = download(i+1)
videos = parse(html)
for item in videos:
print item['title'].encode('utf-8')

若有收获，就点个赞吧

Shell 常用命令
linux 创建新用户
创建新用户
useradd -m -d /PATH/TO/FOLDER USERNAME

删除新用户
userdel [OPTIONS] USERNAME

Use the -r (--remove) option to force userdel to remove the user’s home directory and mail spool:
userdel -r username

授权sudo 权限
usermod -aG sudo username

Bash is not fully functional for a new user新用户可能刚开始只能使用部分功能，比如说history命令都不行。
解决方法 https://unix.stackexchange.com/questions/25475/bash-is-not-fully-functional-for-a-new-user
chsh -s /bin/bash username


--------------------------
cat tv_online_shopping_jsgf_expand.txt | awk -vOFS='\t' '{print $2,$3}' > tv_online_shopping_two_column.txt
 
 cat types.txt | sort | uniq -c | sort -k1,1nr > types.count.txt
将文件中的多行拼接成一行
awk ' { printf ("%s ", $0)} END {printf ("\n") } '
 
awk中gsub的应用
gsub函数则使得在所有正则表达式被匹配的时候都发生替换
gsub(regular expression, subsitution string, target string);简称 gsub（r,s,t)
（1）文件filename的内容
cat awk_file
1 2 3 $1,200.00
1 2 3 $2,300.00
1 2 3 $4,000.00
（2）去掉第四列的$和,并汇总第四列的和。
awk '{gsub(/\$/,"");gsub(/,/,"");cost+=$4;}END{print "The sum is $"cost > "filename"}' awk_file
gsub函数用空串替换$和,再将结果输出到filename中。
（3）输出结果。
cat filename
The sum is $7500
（4）格式化形式的输出
#awk '{gsub(/\$/,"");gsub(/,/,"");
    if ($4>1000&&$4<2000) c1+=$4;
    else if ($4>2000&&$4<3000) c2+=$4;
    else if ($4>3000&&$4<4000) c3+=$4;
    else c4+=$4; }
    END {printf  "c1=[%d];c2=[%d];c3=[%d];c4=[%d]\n",c1,c2,c3,c4}' awk_file
输出结果：c1=[1200];c2=[2300];c3=[0];c4=[4000]
 
awk 'gsub(/\t/, "," ,$0)' filename.txt
 
cut
1、文件内容查看。
cut fl f2 > f3     将把文件fl和几的内容合并起来。
2、显示行中的指定部分
   -b：仅显示行中指定直接范围的内容；
   -c：仅显示行中指定范围的字符；
   -d：指定字段的分隔符，默认的字段分隔符为“TAB”；
   -f：显示指定字段的内容；
   -n：与“-b”选项连用，不分割多字节字符；
   --complement：补足被选择的字节、字符或字段；
   --out-delimiter=<字段分隔符>：指定输出内容是的字段分割符；
   --help：显示指令的帮助信息；
   --version：显示指令的版本信息。
 
 
find . -type f -name "*.vocab" | xargs grep "变形金刚" | grep  "USER.PERSON_VIDEO"
 
File1删除File2的内容
comm -23file1 file2
 
 
centos关闭 CPU throttling
 /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
powersave
Performance
http://blog.sciencenet.cn/blog-935970-892170.html
echo -n performance > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
http://blog.chinaunix.net/uid-20620288-id-5751294.html
 
 
ffmpeg使用
ls | grep -v "*.txt" | grep -v "*.sh" > templst.txt
cat templst.txt | while read line
do
   echo $line
   #name = echo "$line" | cut -d "." -f 1
   #echo "$name"
   if [ "$line"x = "templst.txt"x ]; then
     continue
   fi
   ffmpeg -i $line -ar 16000 ../`echo "$line" | cut -d . -f 1`.wav
Done
 
查看静态库中的文件
 
示例五查看静态库中的文件
[root@node56 lib]# ar -t libhycu.a
base64.c.o
binbuf.c.o
cache.c.o
chunk.c.o
codec_a.c.o
 
[root@node56 lib]# ar -tv libhycu.a
rw-r--r-- 0/0   7220 Jul 29 19:18 2011 base64.c.o
rw-r--r-- 0/0   2752 Jul 29 19:18 2011 binbuf.c.o
rw-r--r-- 0/0  19768 Jul 29 19:18 2011 cache.c.o
...
rw-r--r-- 0/0   4580 Jul 29 19:18 2011 xort.c.o
[root@node56 lib]#
[root@node56 lib]# nm -s libhycu.a | less
 
Archive index:
Base64Enc in base64.c.o
GetBase64Value in base64.c.o
Base64Dec in base64.c.o
encode64 in base64.c.o
decode64 in base64.c.o
check64 in base64.c.o
test64 in base64.c.o
 
JAVA core dump处理
 jstack -J-d64 $JAVA_HOME/bin/java java.core.32134
gdb -c corefile java
options.semantic_size
 
 
wave格式转换
#! /usr/bin/python
__author__ = 'zhuozhu.zz'
import os
 
def dirlist(path):
    allfile=[]
    filelist = os.listdir(path)
 
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile
 
def main():
    orig_wav_dir = "/home/zhuozhu.zz/ximalaya/raw_wav"
    out_dir = "/home/zhuozhu.zz/ximalaya"
    allfile = dirlist(orig_wav_dir)
    for file in allfile:
        print file + '\n'
        file_name = os.path.basename(file)
        (short_name, ext) = os.path.splitext(file_name)
        file_name = short_name + ".wav"
        print file_name
        command = "ffmpeg -i " + file + " -ar 16000 " + os.path.join(out_dir, file_name)
        os.system(command)
 
if __name__ == '__main__':
    main()


gdb -c core.100721 python
https://www.boost.org/users/history/version_1_66_0.html
tar -xzf download
cd boost_1_58_0
./bootstrap.sh --prefix=/your/install/folder
./b2 ; ./b2 install ; cd ..

//---------------------------------------------------------------
./bootstrap.sh --prefix=/disk4/zhuozhu.zz/tops/boost-1.66 --with-python=python
./b2 define=_GLIBCXX_USE_CXX11_ABI=1 install -j5

Vim 使用
vim——打开多个文件、同时显示多个文件、在文件之间切换
打开多个文件：
1.vim还没有启动的时候：
在终端里输入
vim file1 file2 ... filen便可以打开所有想要打开的文件
2.vim已经启动
输入
:open file
可以再打开一个文件，并且此时vim里会显示出file文件的内容。

同时显示多个文件：
:split
:vsplit

在文件之间切换：
1.文件间切换
Ctrl+6—下一个文件
:bn—下一个文件
:bp—上一个文件
对于用(v)split在多个窗格中打开的文件，这种方法只会在当前窗格中切换不同的文件。
2.在窗格间切换的方法
Ctrl+w+方向键——切换到前／下／上／后一个窗格
Ctrl+w+h/j/k/l ——同上
Ctrl+ww——依次向后切换到下一个窗格中

横向分割显示：$ vim -o filename1 filename2   
纵向分割显示：$ vim -O filename1 filename2  
二、如果已经用vim打开了一个文件，想要在窗口中同时再打开另一个文件：
横向分割显示：
:vs filename  
纵向分割显示：
:sp filename  
其中，vs可以用vsplit替换，sp可以用split替换。
如果finename不存在，则会新建该文件并打开。
三、关闭窗口
关闭光标所在的窗口：
:q  
#或  
:close  
关闭除光标所在的窗口之外的其他窗口：
:only  
关闭所有窗口：
:qa  
四、切换窗口
打开了多个窗口，需要在窗口之间切换时：
ctrl + w w
即按住ctrl键，再按两下w键。
或者ctrl + w <h|j|k|l>
即按住ctrl键，按一次w键，再按一次表示方向的h或j或k或l，则光标会切换到当前窗口的 左｜下｜上｜右 侧的窗口



convert_flac_to_wav
for f in `cat flac.lst`
do
  wav=$(echo $f | sed "s/\.flac/\.wav/g")
  echo "convert $f to $wav."
  sox $f -r 16000 -b 16 -c 1 $wav
done

pytplot-jupyter notebook
import pandas as pdimport matplotlib.pyplot as plt

df = pd.read_csv("pretrain_94_95_norm.log", encoding="GB2312", sep='\t')
print(df)

df.drop_duplicates(subset=['epoch',],keep='last',inplace=True)


#fig = plt.figure(figsize=(12, 6))
fig, axes = plt.subplots(1, 1, figsize=(8, 4))

# axes.plot(df['epoch'], df['loss'], linewidth=1.5, label='raw audio')
# axes.plot(df['epoch'], df['loss_norm'], linewidth=1.5, label='audio with norm')
# axes.set_xlabel('epoch')
# axes.set_ylabel('loss')

axes.plot(df['epoch'], df['accuracy'], linewidth=1.5, label='raw audio')
axes.plot(df['epoch'], df['accuracy_norm'], linewidth=1.5, label='audio with norm')
axes.set_xlabel('epoch')
axes.set_ylabel('loss')
axes.set_title('Audio norm affects pretrain accuracy')
plt.legend(loc='upper left') # 标签位置


print(df)

fig.tight_layout()

plt.savefig("Gravitational_Waves_Original.png")
plt.show()
plt.close(fig)


---------------------------------------------------------------------


import pandas as pdimport matplotlib.pyplot as plt

df = pd.read_csv("pretrain_10p.log", encoding="GB2312", sep='\t')
df.drop_duplicates(subset=['epoch',],keep='last',inplace=True)


df2 = pd.read_csv("pretrain_16p.log", encoding="GB2312", sep='\t')
df2.drop_duplicates(subset=['epoch',],keep='last',inplace=True)

print(df2)

#fig = plt.figure(figsize=(12, 6))
fig, axes = plt.subplots(1, 1, figsize=(8, 4))

# axes.plot(df['epoch'], df['loss'], linewidth=1.5, label='raw audio')
# axes.plot(df['epoch'], df['loss_norm'], linewidth=1.5, label='audio with norm')
# axes.set_xlabel('epoch')
# axes.set_ylabel('loss')

axes.plot(df['epoch']-574, df['accuracy'], linewidth=1.5, label='4k audio data')
axes.plot(df2['epoch']-574, df2['accuracy'], linewidth=1.5, label='10k audio data')
axes.set_xlabel('epoch')
axes.set_ylabel('accuracy')
axes.set_title('Audio norm affects pretrain accuracy')
plt.legend(loc='upper left') # 标签位置


fig.tight_layout()

plt.savefig("Gravitational_Waves_Original.png")
plt.show()
plt.close(fig)

若有收获，就点个赞吧

docker相关
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

Ssudodockerps
PORTS
COMMAND
CONTAINERID
STATUS
CREATED
NAMES
binbash"
4b9110329dfd
7azef210c203
Up3days
loving-bassi
3daysqgo
/bin/bash"
qe95qecooga5
ca48bbbb5d6f
10daysago
up10days
zhuozhu
bin/bash"
q036dbo34b21
d29c0712d5f4
10daysogo
zhuozhu.zz
Exitedoiedaysag
Q036db034b21
binbash"
8f556958d818
ExitedO)10di
11daysdgo
fairseg
7bin/bash
qfO5a3843307
3monthsogo
reg.docker.ababal.comu
stoic_mor'se
/bin/bash"
reg.docker.alibaba-inc.comjiling/3
ccqf69cZd450
ExitedO
suspicious-poitras
12monthsogo
bin/bash
5d4c5aa6cb90
12monthsogo
reg.docker.alibaba-inc.comjilig/
Exitedo
peaceful-wescoff
Exited015months
/bin/sh-c
245q8b889056
prickly-wiiliams
apt-get
15monthsogo
defde79e845c
Exited010months
bin/bash"
48cdc084d9fa
17.monthsogo
drunkrosalind
eg.dockeratbabat
Qgo
f32569cq7d92
bin/bash"
ExitedQ10months
mmdnn/mmdnn.cpu.smoll
backstabbing-hodgkin
20monthsogo
5090
20months0go
pbin/bash
tF12
f9eb4a8453c4
Exited(010m
6906/tcp,8888/top
tensorflon/tensorflon:1.4.0-deve-g
monthsogo
3645f854e41c
bin/bash"
2yearsogo
ExitedO2yearsc
nvidig/cudo:8.0-cudnn6-devel-ubuntu14.4
base



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
1创建完成之后可以查看详细信息
    $ docker volume inspect my-vol
   可得信息如下：
  


   注意这个Mountpoint所对应的目录就是我们用来主机和容器进行文件传输的目录。

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

1docker export 和 docker save ：    保存docker镜像到本地tar文件
   docker import 和 docker load ：    从本地tar文件加载docker镜像


Steps For Committing Changes to Docker Image
Step 1: Pull a Docker Image
To illustrate how to commit changes, you first need to have an image to work with. In this article, we work with the latest Ubuntu image for Docker. Download the image from Docker’s library with:



If you recheck the available images, you will see the Ubuntu image:



Copy IMAGE ID for later use.


Step 2: Deploy the Container
Add the IMAGE ID to the command that will create a container based on the image:

The –it options instruct the container to launch in interactive mode and enable a terminal typing interface. Upon executing the command, a new container launches and moves you to a new shell prompt for working inside of it.





Step 3: Modify the Container
Now that you are in the container, you can modify the image. In the following example, we add the Nmap software for network discovery and security auditing:

The command will download the Nmap package and install it inside the running container.



You can verify the installation by running:

The output shows you that Nmap version 7.60 is installed and ready to use.



Once you finish modifying the new container, exit out of it:

Prompt the system to display a list of launched containers:

You will need the CONTAINER ID to save the changes you made to the existing image. Copy the ID value from the output.





Step 4: Commit Changes to Image
Finally, create a new image by committing the changes using the following syntax:

Plain Text
复制代码
1
sudo docker commit [CONTAINER_ID] [new_image_name]
Therefore, in our example it will be:

Plain Text
复制代码
1
sudo docker commit deddd39fa163 ubuntu-nmap
Where deddd39fa163 is the CONTAINER ID and ubuntu-nmap is the name of the new image.




Your newly created image should now be available on the list of local images. You can verify by checking the image list again:

Plain Text
复制代码
1
sudo docker images




1. 登录阿里云Docker Registry

Plain Text
复制代码
1
$ sudo docker login --username=petty_cash registry.cn-hangzhou.aliyuncs.com
用于登录的用户名为阿里云账号全名，密码为开通服务时设置的密码。
您可以在访问凭证页面修改凭证密码。

2. 从Registry中拉取镜像

Plain Text
复制代码
1
$ sudo docker pull registry.cn-hangzhou.aliyuncs.com/docker_workspace_zz/fairseq-wav2vec:[镜像版本号]

3. 将镜像推送到Registry

Plain Text
复制代码
1
$ sudo docker login --username=petty_cash registry.cn-hangzhou.aliyuncs.com$ sudo docker tag [ImageId] registry.cn-hangzhou.aliyuncs.com/docker_workspace_zz/fairseq-wav2vec:[镜像版本号]$ sudo docker push registry.cn-hangzhou.aliyuncs.com/docker_workspace_zz/fairseq-wav2vec:[镜像版本号]
请根据实际镜像信息替换示例中的[ImageId]和[镜像版本号]参数。

4. 选择合适的镜像仓库地址
从ECS推送镜像时，可以选择使用镜像仓库内网地址。推送速度将得到提升并且将不会损耗您的公网流量。
如果您使用的机器位于VPC网络，请使用 registry-vpc.cn-hangzhou.aliyuncs.com 作为Registry的域名登录。

5. 示例
使用"docker tag"命令重命名镜像，并将它通过专有网络地址推送至Registry。

Plain Text
复制代码
1
$ sudo docker imagesREPOSITORY                                                         TAG                 IMAGE ID            CREATED             VIRTUAL SIZEregistry.aliyuncs.com/acs/agent                                    0.7-dfb6816         37bb9c63c8b2        7 days ago          37.89 MB$ sudo docker tag 37bb9c63c8b2 registry-vpc.cn-hangzhou.aliyuncs.com/acs/agent:0.7-dfb6816

使用 "docker push" 命令将该镜像推送至远程。

Plain Text
复制代码
1
$ sudo docker push registry-vpc.cn-hangzhou.aliyuncs.com/acs/agent:0.7-dfb6816


Plain Text
复制代码
1
2
3
4
sudo docker run -it --name wenet_test --shm-size 32G -v /home/zhuozhu.zz:/workspace/zhuozhu.zz reg.docker.alibaba-inc.com/algorithm/quake:torch-1.8.2-cuda111-centos7-train-v2.0 bash
sudo docker exec -it f938098307dd /bin/bash
sudo docker commit asr reg.docker.alibaba-inc.com/docker_workspace_zz/espnet-ctc-att-cuda11-quake:v0.1
sudo docker push reg.docker.alibaba-inc.com/docker_workspace_zz/espnet-ctc-att-cuda11-quake:v0.1

dot 画图及中文乱码解决
sudo yum install graphviz
fstdraw origin.fst | dot -Tpng -Gdpi=2500  -oorigin.png
dot -Tpdf xxx.dot -o xxx.pdf
 
Graphviz 中文乱码
安装字体
yum install cjkuni-ukai-fonts
再次运行
dot graph.gv -Tpng -o image.png
Train non-backoff LM,
一、利用 LM training text， 直接train non-backoff LM。
用SRILM ngram-count 的 -gt1max 0 -gt2max 0 -gt3max 0 -gt4max 0 -gt5max 0 就可以了train non-backoff LM。
Setting -gtNmax to 0 would disable discounting.
 
二、利用 backoff LM， convert 成 non-backoff LM， 而不是只是把BOW 都扔掉。
1、先把每个order的ngram 从 你的backoff LM里抽出来， 比如所有的trigram， 存在一个文件里。
2、用 SRILM 的ngram-rescore function， run ngram -debug 1 -order 3 -lm $backoff_lm -rescore $trigram_file, 它会用这个backoff LM去compute每个ngram 的probability.

prune LM
ngram -lm tiangong.arpa -order 4 -prune 1e-7 -prune-lowprobs -write-lm tiangong.order4.1e-7.arpa

arpa format lm to fst
/home/zhuozhu.zz/kaldi/egs/wsj/s5/utils/format_lm.sh
arpa2fst --disambig-symbol=#0 /home/zhuozhu.zz/tiangong.order4.1e-7.arpa $out_dir/G.fst
 
./fstcompile --isymbols=/home/zhuozhu.zz/tiangong.words.txt --osymbols=/home/zhuozhu.zz/tiangong.words.txt /home/zhuozhu.zz/select_empty.fst.txt | ./fstarcsort --sort_type=olabel | ./fstcompose - /home/zhuozhu.zz/tiangong.order4.1e-7.fst > empty_words.fst
 
./fstinfo ~/empty_words.fst | grep cyclic | grep -w 'y' && echo "Language model has cycles with empty words"

安装Opencc
在centos中，直接使用yum install opencc是不够的，使用opencc会提示没有这个命令。yum search opencc一下，发现有个opencc-tools安装之，使用opencc  -i wiki_00 -o wiki_chs -c zht2zhs.ini命令，果断成功！
