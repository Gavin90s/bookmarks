### 判断字符串是否为纯英文

```
def str_only_contain_english(str):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    str = str.strip()
    if len(str) <= 0 and str == "'":
        return False

    for ch in str:
        if ch.isspace() or ch == "'" or ch == "," or ch == "?" or ch == "!" or ch == ":":
            continue

        if ord(ch) not in range(97, 122) and ord(ch) not in range(65, 90):
            return False
    return True
```

### 判断字符串是否只有汉字
```
def str_only_contain_hanzi(str):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in str:
        if u'\u4e00' <= ch <= u'\u9fff' or ch == ' ' or ch == '．':
            continue
        else:
            return False

    return True
```
    
### 字符串正在匹配
```
p1 = re.compile(r'[《](.*?)[》]', re.S)  # 最小匹配
#p2 = re.compile(r'[(](.*)[)]', re.S)   #贪婪匹配
name = re.findall(p1, raw_name)
```

### 字符串替换
```
re.sub(r"第(.*?)集|第(.*?)季", "", raw_name)
```

### 字符串替换
```
regrex_pattern = r"\(.*?\)|\{.*?\}|\[.*?\]|\【.*?\】|\（.*?\）|\（(.*?)"
name = re.sub(regrex_pattern, "", raw_name)
```

### 字符串替换
```
query = re.sub(' +', ' ', query)
```
