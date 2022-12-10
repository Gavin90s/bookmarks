# Vim 使用

#### vim打开多个文件、同时显示多个文件、在文件之间切换
```
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
```
