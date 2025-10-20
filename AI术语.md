visual grounding, 视觉定位任务
它的核心任务是根据给定的文本描述，在图像中定位特定的区域。
Grounding, the ability to accurately locate and interact with specific GUI
elements, is critical for actions like clicking or dragging

Video-MME（Video Multi-Modal Evaluation）是 首个面向多模态大模型（MLLM）的“全频段”视频理解基准，
由腾讯优图、中科大等团队在 2024 年 5 月提出。它通过 900 支 11 秒～1 小时不等的开放域视频、
2700 道人工精标的多项选择题以及配套字幕/音频，全面考察模型在 时序、感知、推理、跨模态对齐 等维度的能力。

Self-consistency prompting（自一致性提示）
是一种无需训练、零成本集成的解码策略，用来让大语言模型在复杂推理任务中“自己检查自己”，从而提高最终答案的准确率与稳定性。核心做法可以概括为三句话：
- 对同一个问题、同一套 Chain-of-Thought 提示，用温度>0 的采样解码让模型独立地生成多条推理路径（通常 5–40 条）。
- 把这些路径得到的最终答案收集起来，用多数投票（majority vote）选出出现次数最频繁的答案。
- 将该“最一致”答案作为最终输出，并可视投票比例给出置信度。
