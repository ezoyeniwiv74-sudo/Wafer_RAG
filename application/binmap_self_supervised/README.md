# Bin Map 无标签自监督适配

本目录训练 DINOv2-Large 的 Bin Map 专用 LoRA 适配器，不读取任何类别标签，
也不会覆盖原始 DINOv2 权重。训练数据默认使用 `data/YEDN` 中的
`DefectMapPost` 图片；每次从所有缺陷位置中随机取局部区域，因此不会只学习晶圆中心。

输出文件：

- `best_adapter.pt`：验证集检索间隔最好的 LoRA；
- `last_adapter.pt`：最后一轮 LoRA；
- `training_history.json`：训练前基线和每轮无标签检索指标；
- `resolved_config.json`：完整训练配置。

训练目标是局部图像检索，不是缺陷类别分类。SEM 推理时不启用该适配器。
