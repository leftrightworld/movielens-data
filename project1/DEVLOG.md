# Project 1 开发日志 / Dev Log

课程 Project 1（Technical Review：MF-BPR vs LightGCN）的工作记录，按天整理。
供组员了解项目来龙去脉；英文正式文档见 `README.md` 与 `report/report.pdf`。

---

## 2026-09-19：数据集、方法选型、模型实现、全部实验

**数据集准备**
- 下载 MovieLens-1M 并转成分析友好的 CSV（评分加 `datetime` 列、电影拆出 `year`、用户的年龄/职业代码翻译成可读标签），建立本 GitHub repo 托管，Colab 可通过 raw 链接直读。
- Amazon Reviews 2023 初选 **All_Beauty** 品类并转好 CSV。但做协同过滤预处理时发现**致命问题**：该品类绝大多数用户只写过一条评论，标准 5-core 过滤后 69 万条交互塌缩到仅 2,535 条 / 356 用户，完全无法支撑实验（2-core 也只剩 5.7 万条且图结构极碎）。
- 改选 **Video_Games** 品类（460 万条评论）：5-core 后保留 81.5 万交互 / 9.5 万用户 / 2.6 万商品，交互量与 ML-1M 同量级但密度差 **144 倍**（4.84% vs 0.034%）——正好构成"稠密 vs 稀疏"的对照实验设计。

**方法选型**（结合 project 要求 PPT 与课程大纲）
- 方法一 **MF-BPR**（课内经典基线）+ 方法二 **LightGCN**（连接课程前半段 GNN 与后半段 RecSys 两个主题；L=0 时精确退化为 MF，构成天然干净的消融对比）。
- 评测协议：隐式反馈、5-core、每用户 80/10/10 随机切分（seed 42）、全量排序、Recall/NDCG@10/20。

**实现**（纯 PyTorch 从零写，共约 100 行，见 `src/models.py`）
- 踩坑 1：Apple MPS 不支持稀疏矩阵乘 → LightGCN 传播改用 gather + `index_add` 边表实现，CUDA/MPS/CPU 通用。
- 踩坑 2：LightGCN 每步全图传播太慢（M2 上 110 秒/epoch）→ batch 加大到 65536 + lr 3e-3，传播次数减少 9 倍，12 秒/epoch，收敛曲线不变。
- 负采样用固定种子的 NumPy 生成器（CPU），实现跨设备逐位可复现：MF 在 M2(MPS) 与 L4(CUDA) 上测试指标完全一致。

**算力**
- 先在本地 M2 起 18-run 实验队列；随后新开 EC2 `fengyuan-gpu`（g6.xlarge / L4，us-east-2），全部实验迁移到 GPU 重跑（~1.5h），本地队列停止。实例用完即 stop。

**重要的严谨性修正：训练预算公平化**
- 发现多个 LightGCN 配置在 300-epoch 上限处仍在上升（被截断）。300 epoch 预算下层数消融呈现"L=2 峰值、L=4 变差"的经典"过平滑"形状；把所有被截断配置延长到 **600 epoch** 重跑后，"变差"完全消失——**深层传播只是收敛慢，预算不公平会伪装成过平滑**。这一发现写入报告 4.3 节。

**主结果**（test NDCG@10，d=64）
| 数据集 | MF-BPR | LightGCN(L=3) | 提升 |
|---|---|---|---|
| MovieLens-1M | 0.2456 | 0.2605 | +6.1% |
| Video Games | 0.0344 | 0.0494 | **+43.5%** |

**分析发现**
- 增益集中在低活跃用户：ML-1M 最冷分桶 +14.6% → 最活跃 +4.0%（单调递减）；case study 中 476 交互的重度用户 MF 反而命中 7/10 vs LightGCN 4/10。
- 流行度不对称：稠密数据上 LightGCN 降低流行度偏差（覆盖率 51%→61%）；稀疏数据上相反（62%→41%）——+43.5% 中有"押注爆款"的成分，是候选的改进方向。
- 维度扫描：LightGCN d=16（参数量 1/4）在两个数据集上均胜过 MF d=64。

---

## 2026-09-20：All_Beauty 清除、交付物拆分、repo 重组

**All_Beauty 彻底移除**（决定：报告与 repo 不留任何痕迹）
- 删除 `amazon-beauty-2023/` 数据与 notebook/README 中全部相关内容；实验数据源替换为 `amazon-vgames-2023/`（ratings 拆两个分卷绕开 GitHub 100MB 单文件限制 + products 表）。
- 关键保障：改造 `data_prep.py` 直接从 repo 内 CSV 重建数据集，并**逐位比对验证**重建的 train/valid/test 与全部已跑实验所用完全一致——所有结果无需重跑。

**交付物拆分**（此前 notebook 与详细论述混在一起）
- `project1_report.ipynb` 瘦身为**代码集合**：src 源码内嵌 + 一两行说明 + 简短结论。
- 新增 `report/report.tex` → **8 页正式英文报告**（Abstract/Intro/Methods/Experiments/Discussion/Conclusion），嵌入全部 6 张实验图；4 张表格由 `report/gen_tables.py` 从 `results/*.json` 自动生成，杜绝手抄。
- 应要求补充**训练 loss 曲线**（notebook 5.3 节 + 报告 Figure 3）：MF 训练 loss 压得更低（VG 上近乎记住训练集）但测试更差——传播是结构性正则化的直接证据。

**Repo 重组**（目标：外人可轻松读懂）
- MovieLens CSV 收进 `movielens-1m/`；四级 README（根目录地图 + 两个数据集字段表/Colab 代码 + project1 结构与复现指南）。
- 补上传此前仅在本地的 4 个 `*_emb.npz`（主实验 embedding + top-K，分析 cell 依赖）。
- 补齐数据转换脚本 `movielens-1m/convert.py`、`amazon-vgames-2023/convert.py`，代码链条完整可追溯：原始数据 → CSV → npz → 训练 → 图表 → 报告。

---

## 2026-09-21：报告修复与讲解、方法论讨论

- 修复 `report.pdf` 交叉引用显示 "Figure ??" 的 bug（pdflatex 需两遍编译），重新推送并同步桌面 `project1-deliverables/`。
- 撰写并发布中文长文讲解（HTML server 免密页，13 节 + 6 图，个人学习材料，不入本 repo）。
- **方法论讨论记录**：
  - 随机 80/10/10 切分 vs 时间切分：对本报告（方法相对比较、与文献同协议）不构成严重问题；但绝对数字不可类比线上，且"未来流行度泄露"可能让押爆款的 LightGCN 在 VG 上额外占便宜——+43.5% 换时间切分可能缩水。两数据集均有时间戳，`data_prep.py` 加 `--temporal` 开关即可做稳健性验证（候选加分项）。
  - Project 要求核实：方法数是"**至少 2 个，多于 2 个受鼓励**"。候选追加：Popularity 零训练基线（给流行度分析提供锚点，成本极低）、ItemKNN（补全"启发式→浅层学习→图学习"谱系）、以及从流行度发现长出来的 LightGCN 去偏扩展（对应 rubric 的 "improve the methods"）。

## 待办
- [ ] Popularity 基线加入 `run_all.sh` 并进主表
- [ ] （可选）ItemKNN
- [ ] （可选）时间切分稳健性实验
- [ ] 创新扩展：稀疏场景 LightGCN 流行度去偏
- [ ] 提交打包：`group_XX_report.pdf` 改组号重编译；code.zip 去除一切外部链接（notebook 数据路径已是相对路径）
