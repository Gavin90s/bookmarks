# [Goal Alignment in LLM-Based User Simulators for Conversational AI](https://arxiv.org/pdf/2507.20152)
针对 LLM 用户模拟器在多轮对话中无法持续遵循用户目标的错位问题，
提出 UGST(User Goal State Tracking) 框架将目标分解为画像、策略、任务、要求、偏好五类可动态追踪的子组件，
并通过 "UGST用于推理时引导→冷启动 SFT→GRPO 强化学习（通过UGST算reward)" 三阶段训练，使 8B 小模型的目标对齐能力追平甚至超越 70B 大模型，平均成功率最高提升 14.1%。
