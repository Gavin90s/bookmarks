### pickle 读写二进制文件
```
import pickle

file_handler = open(r'nlp_gpt3_text-generation_2.7B/.msc','rb')
model_info = {'Path': 'model/mp_rank_00_model_states.pt', 'Revision': 'ae70879a590d484398bdeed1d68e5255c0afe65f'}

info = pickle.load(file_handler)
# info.append(model_info)
# print("info ", info)

file_handler = open(r'nlp_gpt3_text-generation_2.7B/.msc', 'wb+')
pickle.dump(info, file_handler)
```

### pytorch 中的 @ 和 * 运算符
@ 和 * 代表矩阵的两种相乘方式：
@ 表示常规的数学上定义的矩阵相乘；
* 表示两个矩阵对应位置处的两个元素相乘
```
import torch
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[2, 1], [4, 3]])
print("x_shape", x.shape)
print("y_shape", y.shape)
 
c = x@y
print("c_shape", c.shape)
print(c)
 
 
# 结果： 
x_shape torch.Size([2, 2])
y_shape torch.Size([2, 2])
c_shape torch.Size([2, 2])
tensor([[10,  7],
       [22, 15]])
```
