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
\* 表示两个矩阵对应位置处的两个元素相乘
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

### partial 的用法
```
Partials can be used to make new derived functions that have some input parameters pre-assigned

To see some real world usage of partials, refer to this really good blog post here

A simple but neat beginner's example from the blog, covers how one might use partial on
re.search to make code more readable. re.search method's signature is:

search(pattern, string, flags=0)

By applying partial we can create multiple versions of the regular expression search
to suit our requirements, so for example:

is_spaced_apart = partial(re.search, '[a-zA-Z]\s\=')
is_grouped_together = partial(re.search, '[a-zA-Z]\=')

Now is_spaced_apart and is_grouped_together are two new functions derived from re.search
that have the pattern argument applied(since pattern is the first argument in the re.search method's signature).

The signature of these two new functions(callable) is:

is_spaced_apart(string, flags=0)     # pattern '[a-zA-Z]\s\=' applied
is_grouped_together(string, flags=0) # pattern '[a-zA-Z]\=' applied

This is how you could then use these partial functions on some text:

for text in lines:
    if is_grouped_together(text):
        some_action(text)
    elif is_spaced_apart(text):
        some_other_action(text)
    else:
        some_default_action()

You can refer the link above to get a more in depth understanding of the subject,
as it covers this specific example and much more..
```

#### cprint使用
cprint 是一个 Python 库，用于在终端中打印彩色文本。它提供了一种简单的方法来增强终端输出的可读性和视觉效果。
```
# 打印黄色文本
cprint("This is yellow text", color='yellow')

# 打印带有背景色的文本
cprint("This is text with a red background", color='white', bg='red')

# 打印加粗文本
cprint("This is bold text", color='blue', attrs=['bold'])
```

