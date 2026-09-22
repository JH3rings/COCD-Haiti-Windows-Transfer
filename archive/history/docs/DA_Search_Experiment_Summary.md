# Teacher 双轨学习向单轨 Student 迁移搜索能力：实验总结

## 1. 要验证的科学问题

本实验不是要证明 Student 在推理时仍然需要双轨输入，也不是要证明 Teacher policy 可以原样固定给 Student 使用。

真正要验证的问题是：

> Teacher 在训练阶段同时看到 target 和 counter，能否学习出有用的证据搜索机制；Student 继承该搜索模块后，只使用单轨 target，并经过单轨监督继续训练，最终能否超过没有搜索模块的单轨基线？

因此，Teacher 的 counter 只存在于训练阶段；最终 Student 的部署输入只有 target。

## 2. 模型和训练流程

### T0：普通双轨 Teacher

- 共享 ConvNeXt-Tiny + 三层 FPN。
- target 和 counter 在每个 FPN 层通过普通 `1x1 Conv([T,C])` 融合。
- 没有 Deformable Search。
- 训练损失是两个轨道输出的 BCE 平均值。

### T1：双轨 Teacher + Search

- policy 根据 target 特征和双轨融合特征预测搜索位置和权重。
- policy 使用 target+counter context，但 DA 采样值始终来自 target memory。
- 3 个 FPN level，每个 level 4 个采样点。
- 输出包含双轨主路径和 target-only auxiliary 路径。
- 损失：`L_T1 = L_dual + 0.10 * L_aux`。

### S0：继承 Teacher 后继续单轨训练的 Student

- 从 T1 继承 backbone、target policy、SearchWrite 和 target projection。
- 去掉 counter、dual fusion 和 Teacher 输入。
- 之后继续只用 target 和 GT 训练，所有 Student 参数可以适应性更新。
- 损失：`L_S0 = L_seg`，不使用 logits KD，也不使用额外的 feature KD。
- 最终推理只输入 target，因此 S0 是真正的单轨模型。

## 3. 固定训练协议

- seed：42
- batch size：16
- Adam，学习率 `5e-5`
- 最大 100 epoch
- patience=2
- 没有 validation，使用合并 train
- test AUPRC 用于开发期监控和 best checkpoint 选择
- 所有模型使用同一个 test split 和同一个阈值 0.5

## 4. 关键结果

| 模型 | 推理输入 | Overall AUPRC | Overall IoU | G10 AUPRC |
|---|---|---:|---:|---:|
| SO 旧单轨基线 | target | 0.879639 | 0.633477 | 0.888281 |
| T0 普通双轨 Teacher | target+counter | 0.908018 | 0.671684 | 0.918080 |
| T1 双轨 + DA Teacher | target+counter | 0.909764 | 0.677758 | 0.920059 |
| S0 Teacher-initialized target-only Student | target | 0.896545 | 0.655154 | 0.903567 |

S0 相比 SO：

- Overall AUPRC：`+0.016906`
- Overall IoU：`+0.021677`
- G10 AUPRC：`+0.015286`

S0 的 best epoch 是 19，epoch 21 后连续两次没有提升而停止，说明在本协议下已达到操作性收敛。

## 5. 这些结果支持什么

当前结果支持以下命题：

> 双轨 Teacher 在训练阶段利用 counter 学到的搜索模块，可以被 Student 继承；Student 继续进行单轨 GT 训练后，最终形成只依赖 target 的单轨模型，并且整体指标超过旧的无搜索单轨 SO 基线。

这是一个端到端的 Teacher-to-Student 单轨部署结果。

SO 没有搜索模块是正确的：它代表部署时没有搜索能力的旧单轨基线。S0 不是把 SO 改成另一个模型，而是验证“继承 Teacher 搜索模块后，单轨 Student 能否变好”。

## 6. 不应该过度声称的内容

本实验不声称：

- Teacher policy 不经 Student 适应就可以直接套用；
- 所有提升都能被严格分解成 DA search 单独贡献；
- 固定 Teacher policy 直接套用到 S0 的 oracle probe 可以提升。

Oracle probe 将 Teacher policy 直接套到 S0 上：

- S0 normal AUPRC：0.896545
- Teacher-policy oracle AUPRC：0.895214

这只说明“固定 Teacher policy 直接迁移”没有通过，不否定“继承后继续单轨训练形成自己的搜索 Student”这一主命题。

## 7. 建议给其他 AI 审核的问题

请重点审核以下四点：

1. S0 是否仍然是严格的 target-only 单轨推理模型？
2. Teacher 初始化后继续训练，是否仍然符合 Teacher-to-Student transfer 的方法定义？
3. SO 无 search、S0 继承 search 的比较，是否足以支持端到端部署结论？
4. oracle 失败是否只否定固定 policy 迁移，而不否定 S0 的继承后适应结果？

## 8. 审核通过后的下一步

如果其他 AI 认可上述解释，下一步不应继续无目的地堆模块，而应：

1. 锁定 T1→S0 checkpoint、协议和结果；当前 Student 只保留 S0。
2. 生成正式的 T0/T1/SO-reference/S0 主结果表。
3. 如需论文级稳健性，再按同一协议做一个独立 seed 的确认运行；不改变模型和 loss。
4. 将结论表述为“Teacher-initialized search improves target-only single-orbit deployment”，不要表述为“固定 Teacher policy 直接迁移成功”。

完整机器可读结果见 `experiments/windows_main/da_search/final_report.json`。
