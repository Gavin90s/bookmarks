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
