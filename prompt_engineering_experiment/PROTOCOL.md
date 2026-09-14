# P0/P1/P2 视觉线索 Prompt 实验协议

## 1. 状态与范围

- `protocol_id`: `visual-cue-prompt-pilot-v1`
- 状态：`exploratory_pilot`
- 冻结日期字段：2026-09-05
- 推断单位：图像
- 评估图像：16
- 模型：GPT-5.6、GPT-4o、Qwen 3.6 Flash
- 条件：P0、P1、P2
- repeats：3
- 温度：0
- schedule seed：20260905

本协议描述计划和解释边界，不声明 live-run 已执行或任何效果已观察到。机器可读设置以 `experiment_config.json` 为准；实际运行身份以 `protocol_manifest.json` 的完整 SHA-256 为准。

## 2. 研究问题与假设

主要研究问题：blind、非重叠视觉 cue 所形成的决策规则，是否能提高模型单标签输出与既有参与者多标签 endorsement 的对应程度，并且这种变化是否超过一般结构化决策提示产生的变化？

冻结的主要模型是 GPT-5.6，主要比较是 P2 对 P0。方向性研究假设是 P2 的图像宏平均预期参与者 endorsement 高于 P0。P1 用于机制解释；GPT-4o 与 Qwen 3.6 Flash 用于探索模型间可迁移性，不扩充主要检验的样本量。

## 3. Prompt 条件

### P0：retained baseline

P0 原样导入已有 baseline。标签定义、主 moral concern 选择规则、confidence 量表和只返回三个 JSON 字段的要求不变。

### P1：structured-decision control

P1 在 baseline 的 `Confidence:` 段前插入六步内部流程：中性盘点可见证据、区分观察与解释、比较最佳候选/替代/None、按整幅场景证据选择、拒绝隐含故事、按歧义校准 confidence。它控制“增加结构和审慎步骤”本身的影响，但不是与 P2 完全等长、完全等内容的 placebo。

### P2：visual-cue-guided intervention

P2 在同一位置插入 cue-guided 六步流程。它要求按诊断性而非视觉显著性加权直接行动、后果、背景、表情姿态、文字、群体构成和可见性限制；要求独立 cue 汇聚、整幅场景一致性、文字字面约束、身份中立、遮挡不重建及对最含混 cue 的反事实折扣。

P2 不包含评估图 ID、训练示例、预期答案或 cue→label 固定映射。三条件继续使用同一标签集和同一输出 schema。

## 4. Blind 推导与泄漏边界

P2 仅允许使用 `manifests/cue_discovery_cases.csv` 中的五个非重叠视觉干预家族：表情编辑、文字消融、人物区域遮挡、手势遮挡和背景替换。其 base image IDs 与 16 张评估图不重合。

三项存在评估重叠风险的家族 `1-1`、`7-2`、`16-2` 不得用于规则形成。规则作者不得查看 human/survey annotation、匿名 endorsement target 或已有结果表。五个允许案例只能用于发现通用证据纪律，不得产生图像专用规则或预期标签。

非重叠降低的是直接答案泄漏风险，不等于消除了概念重叠、研究者自由度或对同一视觉领域的间接适配。当前代码冻结两个 manifests 的哈希，但访问隔离主要仍是程序外的研究治理要求。

建议保留以下审计记录：

1. 规则作者可访问资料清单及完成时间；
2. P0/P1/P2 prompt 快照及哈希；
3. target 生成者与规则作者的角色分离，或至少 target 在 prompt 冻结前未被查看的声明；
4. 首次 live 请求的时间和 `protocol_sha256`；
5. 所有 refreeze 的原因。首次 live 后的实质改动必须使用新协议 ID。

### Derivation audit

P1/P2 由隔离分析起草。允许材料仅包括：retained baseline prompt；三个模型 CSV 中五个指定 base IDs 的响应；console transcript 中五个指定 variants 的响应片段；以及对应原图和处理图（如存在）。分析过程禁止读取任何 human/survey annotation、16 图评估结果或论文结果表。prompt 文本本身没有保留下表的图像 ID、模型答案或案例专用映射。

| 允许的 base / variant | 干预 | 推导时可见的主要 confound |
| --- | --- | --- |
| `16-1` / `16-1upset.jpg` | distressed-expression edit | 本地未找到对应处理图，无法视觉核实改动范围；一个模型的 variant 请求失败，只有两个完整模型配对。 |
| `34-1` / `34-1text.jpg` | text ablation | 同时移除了多处文字及其视觉载体和显著性；原始监控场景本身支持多种解释。 |
| `g-7` / `g-7women.jpg` | participant-region occlusion | 遮挡同时删除了人口构成、面孔、身体语言、互动数量和中央视觉焦点，不能归因于单一身份 cue。 |
| `12-1` / `12-1hand.jpg` | ritual-gesture occlusion | 多只手及部分姿态线索同时被遮挡，并引入明显 blur artifact；不是严格的单手势消融。 |
| `32-2` / `32-2bg.jpg` | background replacement | 大部分场景、光照和事件语义一起变化；保留的前景外观仍可支持多个故事。 |

这些观察只支持诸如“优先直接证据”“含混 cue 需要独立汇聚”“文字和身份不能作为捷径”“遮挡内容不得重建”“背景只有在直接呈现事件或后果时才是强证据”等通用决策纪律。五个有 confound 的一次性干预和模型响应不能证明某个 cue 导致某个 label，不能确定案例真值，也不能估计 prompt 规则的因果效应。

## 5. 样本、target 与无需新标注者的边界

现有 target 来自 56 条保留参与者记录，共 280 条 image-participant responses，覆盖固定的 16 张图。参与者可以多选标签，因此每个图像-标签的 `endorsement_rate` 是重叠选择率，不是 11 类单标签概率分布。

本 pilot 可以不新增标注者，因为问题被限定为：在 blind prompt 推导后，测试模型输出与一个既有、可匿名聚合的 held-out 参照是否更一致。`prepare_human_targets.py` 原地读取受限 ZIP，只输出 16×11 聚合矩阵和非 respondent-level metadata，不把原始记录送入模型推断。

该理由只支持内部探索，不支持确认性结论。既有样本是为其他研究流程收集并被重复使用的；16 张图是目的性选择而非总体随机样本；没有新的预注册参与者或图像复制样本；模型单标签和人类多标签任务也不完全同构。任何确认性主张都需要新的前瞻性样本。

## 6. 试验设计与请求数

冻结的 provider 设置如下；应在结果中同时核查 `requested_model` 与 provider 返回的 `actual_model`：

| 模型 key | requested model / endpoint | reasoning | image detail |
| --- | --- | --- | --- |
| `gpt56` | `gpt-5.6` / OpenAI Responses | `effort=none` | `high` |
| `gpt4o` | `gpt-4o` / OpenAI Responses | provider 默认 | `high` |
| `qwen36` | `qwen3.6-flash-2026-04-16` / OpenAI-compatible Chat Completions | `enable_thinking=false` | provider 默认 |

每个模型的冻结 schedule 为：

```text
16 images × 3 conditions × 3 repeats = 144 unique trials
```

三个模型合计：

```text
144 × 3 = 432 unique planned trials
```

每个 run 包含 16 个 image blocks，每个 block 内运行三个条件。确定性 Latin rotation 使每个 image-condition 在三个 repeats 中各占据 condition position 1、2、3 一次；各 run 的图像 block 顺序由 seed 确定性打乱。

温度 0 不保证 provider 层完全确定，因此保留三个 repeats。repeats 先在同一图像和条件内汇总，不能当作 48 或 432 个独立统计单位。默认 retry 上限为每 trial 3 次，故实际 API attempts 可高于 432。

## 7. 主要 estimand 与推断

令 `E(i, l)` 为图像 `i` 上参与者对标签 `l` 的 endorsement rate；令 `L(i, c, r)` 为指定模型在图像 `i`、条件 `c`、repeat `r` 选择的标签。定义：

```text
score(i, c, r) = E(i, L(i, c, r))
mean_score(i, c) = mean over the 3 repeats of score(i, c, r)
delta(i) = mean_score(i, P2) - mean_score(i, P0)
primary_effect = mean over 16 images of delta(i)
```

冻结的主要 estimand 是 GPT-5.6 上的 `primary_effect`。每张图贡献相同权重，不按参与者人数、模型 confidence 或标签频率重新加权。

主要不确定性分析为：

- 对 16 个 image-level `delta(i)` 做精确双侧 sign-flip permutation test；
- 以图像为单位有放回重采样 20,000 次，seed 为 20260905，报告 95% percentile bootstrap confidence interval；
- `alpha = 0.05`。

仅当点估计为正、95% 区间完全高于 0 且双侧 `p < 0.05` 时，才可表述为“在本 pilot 的 16 张图内有改善证据”。否则应按点估计和区间报告为更差或不确定；未显著不能证明等效。

主要分析要求 GPT-5.6 的 P0/P2 在每张图均有三个成功且协议一致的唯一 trial。缺失、重复冲突、错误 prompt/image hash 或 actual model 异常不得静默当作 0 分或独立补样；应先修复同一协议下的缺失运行，或把偏离记录为 protocol deviation。

## 8. 次要指标

次要结果用于解释，不改变主要 estimand：

1. `modal_label_plurality_set_match`：先从三个 repeats 得到每个 image-condition 的模型 modal label，再检查它是否属于该图的人类 plurality set。模型 modal 并列时，仅为生成确定性摘要使用 `VALID_LABELS` 的固定 taxonomy 顺序，并必须同时标记 tie；人类 plurality 并列应保留完整集合。
2. `wrong_to_right_and_right_to_wrong`：以 P0→P2 的 image-level modal plurality match 为基准，分别报告不匹配→匹配与匹配→不匹配的图像数，不用净值掩盖反向退化；对 discordant images 的差异另报双侧 exact McNemar p 值。
3. `within_image_label_stability`：主摘要为 modal label count 除以成功 repeat 数，并同时保留 repeat-pair label agreement；报告不稳定图像。它衡量重复性，不等价于与人类一致。
4. `model_reported_confidence`：按 image-condition 汇总模型自报 confidence，并与标签稳定性和 endorsement 分开解释。它不是校准概率。
5. P0↔P1、P1↔P2 及 GPT-4o/Qwen 的同类比较均为次要或探索性。除非另行预注册多重性策略，不应把其中最有利的比较升级为主要发现。

若 P2−P0 为正但 P2 不优于 P1，证据更符合一般结构化决策效应；P2 同时优于 P0 和 P1 才较支持 cue-specific 增益，但仍不能从整包 prompt 操作中识别某一条规则的独立因果作用。

## 9. Mixed-None sensitivity

主分析保留参与者原始选择，包括 3 条同时选择 None 与至少一个 moral label 的 responses。

敏感性分析只从这 3 条 mixed responses 中移除 None endorsement：

- 其他 moral-label selections 保持不变；
- 每张图的 `n_responses` 分母保持不变；
- 主指标改用 `mixed_none_removed_rate` 重新计算；
- 若报告 plurality 类次要指标，应根据 sensitivity counts 重新构造 plurality set，不得沿用主分析的 set。

应并列报告主分析与 sensitivity 的效应、区间、p 值及结论方向是否改变。移除 mixed None 是对问卷语义的替代解释，不是把原记录宣布为错误。

## 10. 冻结、运行和质量门槛

标准顺序为：

1. 完成 config、manifests、P0/P1/P2 和 runner；
2. 运行 `repeatability_env/bin/python prompt_engineering_experiment/freeze_protocol.py`；
3. 在 blind 边界外，由授权人员运行 `repeatability_env/bin/python prompt_engineering_experiment/prepare_human_targets.py --survey-zip ...` 生成匿名 target；
4. 运行 `repeatability_env/bin/python prompt_engineering_experiment/run_prompt_experiment.py --models gpt56 gpt4o qwen36 --dry-run`，核查三个模型的完整计划；
5. 确认密钥、provider 数据政策、预算和输出权限；
6. 完成 live-run，并对同一协议 resume 失败 trial；
7. 校验 432 个唯一计划 trial 的覆盖与偏离；
8. 运行 `repeatability_env/bin/python prompt_engineering_experiment/analyze_prompt_experiment.py --models gpt56 gpt4o qwen36`。

冻结 manifest 覆盖 config、两个 manifests、schedule、runner、三个 prompt 快照和 16 张图像。runner 在创建 API client 前校验冻结输入；响应还必须严格满足三个字段、合法标签、整数 0–100 confidence 和非空 reason。requested/actual model、SDK 版本、时间、token、错误、重试和原始响应均需保留。

匿名 target、target 生成脚本和分析脚本当前不属于 protocol hash 的组成部分。最终报告必须另行记录 target CSV、target metadata 和分析代码版本/哈希；否则冻结只能保证推断侧输入，不能完整保证端到端分析复现。

## 11. 隐私与安全

- Qualtrics ZIP 可能含直接或间接标识信息，只能在获授权的本地环境读取，不得复制到实验目录或发送给模型 provider。
- 匿名化脚本不输出 respondent-level 行，但 metadata 包含源文件名和哈希，分享前仍需审查。
- Live-run 会向 OpenAI 或 Alibaba Cloud endpoint 发送评估图和 prompt。应在运行前确认账户、区域、合同和保留政策；`store=False` 不是跨 provider 的隐私保证。
- `OPENAI_API_KEY`、`DASHSCOPE_API_KEY` 及可选的 `DASHSCOPE_BASE_URL` 只通过受控环境配置，不写入仓库或日志。
- 结果 CSV 和 raw JSON 含绝对路径、provider metadata 与模型自由文本，默认按受控研究数据保存。

## 12. 限制与允许的结果表述

必须同时报告以下限制：

- 只有 16 张目的性选择图像，image-level 推断功效低且对单图敏感；
- 没有新增标注者或独立图像复制，不能声称 out-of-sample generalisation；
- 人类多标签 endorsement 与模型单标签输出并非同一任务；
- 同一参与者可贡献多张图，280 条 responses 不是 280 个独立推断单位；
- P2 仅从五个编辑案例归纳，编辑伪影和小样本都可能影响规则；
- 非重叠约束不能消除概念层面的过拟合或研究者自由度；
- P1 不是与 P2 完全匹配的 placebo，P2 效应不能归因到某个单独句子；
- 温度 0 仍可能有 provider 随机性，模型版本和 endpoint 也会随时间变化；
- plurality match 压缩了 endorsement 强度，并列和 mixed None 会影响摘要；
- 模型 confidence 未经校准；
- endorsement 对应度不是道德正确性、规范正当性或社会共识的证明。

推荐表述是“在冻结的 16-image exploratory pilot 中，P2 相对 P0 的参与者 endorsement 对应度估计为……”。禁止写成“P2 提高了总体准确率”“模型更符合人类道德”或“该 cue 规则已经泛化”。

最终报告至少应给出：协议哈希、target 和分析代码版本、trial 完整性及 failures、主要 effect/95% CI/exact p、P1 与跨模型结果、mixed-None sensitivity、label stability、双向错误转移、所有 ties 和 protocol deviations。
