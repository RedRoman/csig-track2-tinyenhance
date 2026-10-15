# Push4 / stage5 独立训练审计

状态：PASS。仅使用CPU，未初始化CUDA。三条分支各续训500步，6个不可变里程碑均通过；此结论仅证明训练及保存状态一致性，原评分工具的提升须依据独立pilot/full评分。

起点为C7250的514项FP32 LoRA、Adam1250和四类随机状态。固定教师6500；A MUSIQ1.5/reward.15，B MUSIQ2.5/reward.15，C MUSIQ2.5/reward.10，均LR2e-7/CLIP.5/L1 1/gradient.3/LPIPS.1。

独立复核了父权重与resume逐项相等、教师封存SHA及初始迁移、1028个Adam矩迁移、实际训练梯度、每步损失公式、首张case93/crop(60,1399)、跨分支完整采样序列和四类RNG相等、源码及CLI配置SHA、不可变marker所载SHA和inode/mtime。两次里程碑审计均对全部514项FP32参数、raw/resume等值和Adam矩有限性及步数进行了检查。

| 分支 | local | global | Adam | 相对7250参数L2 | raw SHA256 | resume SHA256 |
|---|---:|---:|---:|---:|---|---|
| A | 250 | 7500 | 1500 | 0.054692050 | `2f42290e89af582a532b75e4871cf8fa92fce0d8bbd86c5dc0ebe00286657b70` | `9b286f93797a4160ff3bd38647d2a9b9a509eae6efa0e97b8c3608d4fa888e6d` |
| B | 250 | 7500 | 1500 | 0.061287092 | `70f3d12dbf8bb094ab6d5ed8c1da0d61f39783460201ea3a08b5647714462aab` | `6097e41327e42483d76181e21587b8c4e64f4004552e030753e533cfa0b1c045` |
| C | 250 | 7500 | 1500 | 0.053808256 | `4d6bc029c1c32693e22ec279d651759d96ad1b8bca32a69c7bbf26657ee31f5f` | `8fda7ca1850537297500ec267b79ff7c4d2f9de4263f094fdbf239d50aa5a070` |
| A | 500 | 7750 | 1750 | 0.080313234 | `589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d` | `e240f3e810654e9503ed620a1007649d2dd056c290960b442e220e0a1451a83b` |
| B | 500 | 7750 | 1750 | 0.092331528 | `03dc54bc9c748c0cdfa6889e1f5ac2dd07716b87a6c2fd0601b6c4c1af8ecbb6` | `729d3d8f22858c8cd811d7bbafa97a361a623addb9f5872805dbe970396390a0` |
| C | 500 | 7750 | 1750 | 0.077146694 | `c953cb853a2a98e1563171798af9d4cb5d034957a4d7da444eb1e25fd6db96a3` | `1d10cb6059ef95f38236fe5e5f3b9e1bbbabc4825849198d7bed097288b05d18` |

机器可读完整证据见本目录 `TRAINING_AUDIT_SUMMARY.json`、`initial_cpu_audit.json`、`milestone250_cpu_audit.json`、`milestone500_cpu_audit.json`。审计器及计划均另存并固定SHA。源文件和先前Push3备份未修改。

另完成新recipe-only备份助手的独立静态审阅，无阻断项；此审阅未运行备份、未写入SFS，实际选中模型仍须通过最终selection/full/recipe自身恢复证明和源文件门控。
