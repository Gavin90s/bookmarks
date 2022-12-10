#### pytplot-jupyter notebook

```
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
```

```
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
```
