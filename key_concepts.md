#### DPO(Direct Preference Optimization) 算法
![image](https://github.com/user-attachments/assets/2c69c2d6-dfbc-4d18-bdb5-d4d385c5a832)
![image](https://github.com/user-attachments/assets/bfdfa71f-6754-4c7d-9a6d-ed6dc1333346)
DPO 通过简化了PPO算法，在训练的时候不再需要同时跑4个模型（reward model, ref model, critic, actor），
而是只用跑 actor和 ref 2个模型，甚至由于不再在线采数据，ref model的输出可以预先存下来，训练的时候重复使用。

DPO loss 包括2个部分：reward loss + kl divergence。
