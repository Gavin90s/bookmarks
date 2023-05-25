#### conda 使用
```
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

conda init bash 修复提示符问题
<img width="576" alt="image" src="https://github.com/Gavin90s/bookmarks/assets/8350994/351344f4-df96-47ce-8f59-6dca8e14812a">

```
