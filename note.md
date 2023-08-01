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

#### python fire 库
```
import fire

def hello(name):
  return 'Hello {name}!'.format(name=name)

def main():
  fire.Fire(hello)
不再需要指定hello函数，因为我们调用了fire.Fire(hello)
```

#### oracle WER 
```
计算asr decoding lattice 和hypothesis之间的关系。
it prints out the corresponding WERs in its logging output. This is done by constructing an "edit-distance FST" and composing this with unweighted acceptors constructed from the lattice and from the reference.
```

#### danielpovey主页
http://www.danielpovey.com/publications.html

#### huggingface tranformers安装
```
import error “version `GLIBC_2.29’ not found” 解决方法
conda install -c conda-forge transformers
conda install importlib-metadata
```

#### huggingface tranformers将 tf 模型转化为 pytorch模型
```
https://huggingface.co/transformers/converting_tensorflow_models.html
transformers-cli convert --model_type bert \
                         --tf_checkpoint roberta_tiny_50G_whole_model.ckpt.index \
                         --config bert_config_tiny.json \
                         --pytorch_dump_output pytorch_model.bin
```

#### roberta 代码
https://github.com/Gavin90s/CLUEPretrainedModels/blob/b384fd41665a8261f9c689c940cf750b3bc21fce/baselines/models/roberta/modeling.py#L754
#### bart 代码
https://github.com/huggingface/transformers/blob/master/src/transformers/models/bart/modeling_bart.py

#### 自监督预训练（三）wav2vec 2.0原理剖析
https://blog.csdn.net/xmdxcsj/article/details/115787729

#### CLUECorpus2020
A Large-scale Chinese Corpus for Pre-training Language Model

#### 集团docker地址
https://docker.alibaba-inc.com/#/dockerImage/2955998/detail

#### Scheduled Sampling
```
主要应用在序列到序列模型的训练阶段，而生成阶段则不需要使用。训练阶段解码器在最大化第t个元素概率时，标准序列到序列模型使用上一时刻的真实元素yt−1作为输入。设上一时刻生成的元素为gt−1，Scheduled Sampling算法会以一定概率使用gt−1作为解码器输入。
https://blog.csdn.net/zlrai5895/article/details/84748749
```

#### 查看压缩包内容
```
tar -tf xxxx.tar.gz
```

#### prefix beam search
```
https://blog.csdn.net/weixin_42615068/article/details/93767781
```
#### 解决sort磁盘空间不足
```
export TMPDIR=/tmp
```
#### pytorch 调试
```
import torchsnooper
@torchsnooper.snoop()
```
#### 调研报告｜在线语音识别改进之 RNN-T 训练
https://zhuanlan.zhihu.com/p/146832796

#### 多线程压缩
```
#下载pigz
wget https://download-ib01.fedoraproject.org/pub/epel/7/x86_64/Packages/p/pigz-2.3.4-1.el7.x86_64.rpm
    
#安装pigz    
rpm -ivh pigz-2.3.4-1.el7.x86_64.rpm 

#多线程压缩
tar cf - wenetspeech | pigz -9 -p 32 > /data6/zhuozhu.zz/wenetspeech_8k.tar.gz

#多线程解压
pigz -dc wenetspeech_8k.tar.gz | tar xf -
```

#### gdb 调试
gdb test.bin pid.core

#### VScode c++环境
wget https://vscode.cdn.azure.cn/stable/f4af3cbf5a99787542e2a30fe1fd37cd644cc31f/VSCode-darwin-universal.zip
```
#使用clang-format插件
    
#安装 clang-format, 用code formatter
wget http://releases.llvm.org/8.0.0/clang+llvm-8.0.0-x86_64-apple-darwin.tar.xz
tar xvfJ clang+llvm-8.0.0-x86_64-apple-darwin.tar.xz -C ./clang-format
ln -s /Users/zhuozhu/clang-format/clang+llvm-8.0.0-x86_64-apple-darwin/bin/clang-format /usr/local/bin/clang-format
```

#### opustool安装
```
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
wget https://codeload.github.com/xiph/opus-tools/zip/refs/heads/master
wget https://ftp.osuosl.org/pub/xiph/releases/ogg/libogg-1.3.5.tar.gz

git clone https://github.com/xiph/opusfile.git
git clone https://github.com/xiph/libopusenc.git
git clone https://github.com/xiph/flac.git

opusenc --bitrate 6 --max-delay 10 hkust/dev/data/format.1/20040503_222707_A000687_B000688-B-000391-001021.wav 20040503_222707_A000687_B000688-B-000391-001021.opus
```

#### 查看python包路径
```
import a_module 
print(a_module.__file__)
```
#### 在模型中将某些参数设置为不可学习但保存在模型中
self.register_buffer('my_buffer', buffer) 

```
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
```

#### linux 终端使用技巧
```
https://blog.csdn.net/qq_26399665/article/details/81063211
Ctrl – a ：移到行首
Ctrl – e ：移到行尾
Ctrl – k ：由光标所在位置开始，删除右方所有的字符，直到该行结束。
Ctrl – u ：由光标所在位置开始，删除左方所有的字符，直到该行开始。
Ctrl -a + Ctrl -k 或 Ctrl -e + Ctrl -u 或 Ctrl -k + Ctrl -u 组合可删除整行。
Ctrl-L：进行清屏操作
Ctrl-d 由光标位置开始，往右删除单词。往行尾删
Ctrl – y ：粘贴之前删除的内容到光标后。
粘贴 Shift-Command-V
```

#### 查看.so、.a 文件的函数
nm -C --defined-only ../extra/kaldi/src/base/kaldi-base.a | grep "RandInt"

#### mkv 转 wav
```
ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=nokey=1:noprint_wrappers=1 10027992976_10027992977-03-26-21.mkv
ffprobe -v error -select_streams a:1 -show_entries stream=codec_name -of default=nokey=1:noprint_wrappers=1 10027992976_10027992977-03-26-21.mkv

ffmpeg -i 10027992958_10027992959-03-26-14.mkv -map 0:a:0 xxx.wav
ffmpeg -i 10027992958_10027992959-03-26-14.mkv -map 0:a:1 yyy.wav
```

#### pip 清华源下载
pip install sklearn -i https://pypi.tuna.tsinghua.edu.cn/simple 

#### THOP: 统计 PyTorch 模型的 FLOPs 和参数量
https://blog.csdn.net/yiran103/article/details/97756720

#### python map 用法
```
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
```

#### 在catch中打印完整堆栈
```
catch (Exception e) {
  StringWriter errorsWriter = new StringWriter();
  e.printStackTrace(new PrintWriter(errorsWriter));
  Logger.info("get exception in matchTopicDaEQuan: [%s]", errorsWriter.getBuffer().toString());
  return RetCode.RE_ERROR;
}
```

#### Shell 常用命令
```
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
```

#### awk 使用
```
cat tv_online_shopping_jsgf_expand.txt | awk -vOFS='\t' '{print $2,$3}' > tv.txt
 
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
```
 
#### centos关闭 CPU throttling
```
  /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  powersave
  Performance
  http://blog.sciencenet.cn/blog-935970-892170.html
  echo -n performance > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
  http://blog.chinaunix.net/uid-20620288-id-5751294.html
```
 
 
#### ffmpeg使用
```
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
```

#### 查看静态库中的文件
```
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
```

#### JAVA core dump处理
```
jstack -J-d64 $JAVA_HOME/bin/java java.core.32134
gdb -c corefile java
```
 
#### wave格式转换
```
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
```

gdb -c core.100721 python

#### boost 安装
```
https://www.boost.org/users/history/version_1_66_0.html
tar -xzf download
cd boost_1_58_0
./bootstrap.sh --prefix=/your/install/folder
./b2 ; ./b2 install ; cd ..

//---------------------------------------------------------------
./bootstrap.sh --prefix=/disk4/zhuozhu.zz/tops/boost-1.66 --with-python=python
./b2 define=_GLIBCXX_USE_CXX11_ABI=1 install -j5
```

#### convert_flac_to_wav
```
for f in `cat flac.lst`
do
  wav=$(echo $f | sed "s/\.flac/\.wav/g")
  echo "convert $f to $wav."
  sox $f -r 16000 -b 16 -c 1 $wav
done
```

#### dot 画图及中文乱码解决
```
sudo yum install graphviz
fstdraw origin.fst | dot -Tpng -Gdpi=2500  -oorigin.png
dot -Tpdf xxx.dot -o xxx.pdf
```

#### Graphviz 中文乱码
```
安装字体
yum install cjkuni-ukai-fonts
再次运行
dot graph.gv -Tpng -o image.png
```

#### Train non-backoff LM
```
一、利用 LM training text， 直接train non-backoff LM。
用SRILM ngram-count 的 -gt1max 0 -gt2max 0 -gt3max 0 -gt4max 0 -gt5max 0 就可以了train non-backoff LM。
Setting -gtNmax to 0 would disable discounting.
 
二、利用 backoff LM， convert 成 non-backoff LM， 而不是只是把BOW 都扔掉。
1、先把每个order的ngram 从 你的backoff LM里抽出来， 比如所有的trigram， 存在一个文件里。
2、用 SRILM 的ngram-rescore function， run ngram -debug 1 -order 3 -lm $backoff_lm -rescore $trigram_file, 它会用这个backoff LM去compute每个ngram 的probability.

prune LM
ngram -lm tiangong.arpa -order 4 -prune 1e-7 -prune-lowprobs -write-lm tiangong.order4.1e-7.arpa

arpa format lm to fst
/home/zhuozhu.zz/kaldi/egs/wsj/s5/utils/format_lm.sh
arpa2fst --disambig-symbol=#0 /home/zhuozhu.zz/tiangong.order4.1e-7.arpa $out_dir/G.fst
 
./fstcompile --isymbols=/home/zhuozhu.zz/tiangong.words.txt --osymbols=/home/zhuozhu.zz/tiangong.words.txt /home/zhuozhu.zz/select_empty.fst.txt | ./fstarcsort --sort_type=olabel | ./fstcompose - /home/zhuozhu.zz/tiangong.order4.1e-7.fst > empty_words.fst

./fstinfo ~/empty_words.fst | grep cyclic | grep -w 'y' && echo "Language model has cycles with empty words"

安装Opencc
在centos中，直接使用yum install opencc是不够的，使用opencc会提示没有这个命令。yum search opencc一下，发现有个opencc-tools安装之，使用opencc  -i wiki_00 -o wiki_chs -c zht2zhs.ini命令，果断成功！
```

#### pip 指定安装源
```
pip install fire -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### [大模型对比评测](https://crfm.stanford.edu/helm/v0.2.0/?group=core_scenarios)

#### git LFS 文件pointer丢失的问题
```
Encountered 1 file that should have been a pointer, but wasn't:
        resources/tf-idf/HX11_CN_全量电子版用户手册.review.all.txt
(base) zhuozhu@appledeMacBook-Pro-2 chatgpt-pdf-qa-poc % git lfs fsck --pointers
Git LFS fsck OK
```

#### 大规模中文段落检索数据集DuReader-retrieval
段落检索是从大规模语料库中找出相关段落的任务。随着预训练语言模型的快速发展，稠密段落检索方法的性能取得了质的飞跃，逐步超越了传统的BM25等方法。这种方式能够对查询和候选段落进行语义级别建模，在问答等语义匹配要求高的场景表现更好。我们发布了首个大规模中文段落检索数据集DuReader-retrieval，该语料来源于真实搜索场景，任务难度大，覆盖了真实应用中诸多有挑战的技术问题。

#### 搜索知识对话数据集DuSinc
知识对话指让系统具备利用搜索引擎知识进行开放域对话交互的能力，提升对话的丰富性与知识准确性。生成式对话系统将大量知识存储在模型参数中，能够实现较为流畅的聊天，但在强时效性、细粒度的话题中，表现欠佳。为了应对该挑战，本次竞赛提出了利用搜索引擎实时获取知识然后基于该知识进行对话的任务。我们首次发布了领域开放的搜索知识对话数据集DuSinc，希望参赛系统能够生成合适的搜索Query与知识对话回复。

#### [在任务型对话里面，WOZ是什么意思？](https://zhuanlan.zhihu.com/p/344837663)
（Wizard-of-Oz，中文译名为绿野仙踪，又叫做奥兹国奇遇记）数据集构造方法是一种非常常见的方法，简要来说就是为了实现一个人机对话的语料库，我们让一个人来扮演“机器”（叫做Wizard），另一个来扮演人（human），通过他们的human-human对话为模型提供高质量语料。从经典的ATIS数据集，再到现在multi-domain领域非常热门的数据集系列，New Woz（MultiWoz1.0）、MultiWoz2.0、2.1，2.2都是基于此方法。

#### [CrossWOZ 数据集](https://arxiv.org/pdf/2002.11893.pdf)
CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset

#### multi-hop QA
multi-hop QA, 具体来说，给定一个问题，系统只通过一个文档是无法正确回答问题的，需要根据多篇文档回答一个问题，需要多跳推理。
<img width="503" alt="image" src="https://github.com/Gavin90s/bookmarks/assets/8350994/87c52809-eec3-43e5-8739-e573f9754c9c">

