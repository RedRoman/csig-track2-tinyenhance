# NAFNet32 原生保真蒸馏准备

当前仅完成未训练模型的 CPU 注册与正确性检查，未启动 GPU 或训练。主代理先测固定 640 上下文推理，目标门槛为 512 输入 p10 <12ms。此模型没有已验证的质量或分数。

模型来自官方 NAFNet 架构的独立随机初始化，宽度32、encoder `[1,1,1,28]`、middle1、decoder `[1,1,1,1]`，共17,111,907参数、664参数张量。保留官方版权与论文说明，并随附读取时的原文件 `ORIGINAL_NAFNet_arch.py`；原始源 SHA 为 `01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967`。来源：[官方固定提交](https://raw.githubusercontent.com/megvii-research/NAFNet/2b4af71ebe098a92a75910c233a3965a3e93ede4/basicsr/models/archs/NAFNet_arch.py)。没有使用 G4 架构或权重。

适配只包含独立的纯 Torch 通道 LayerNorm、零输出头初始化与固定原生 tiler。LayerNorm 的 FP16/BF16 统计使用 FP32，CPU FP64 检查保留 FP64。零输出头加官方 residual skip 构成恒等初始化；beta/gamma 的官方零初始化也保留，因此训练首步部分参数梯度为零是预期行为。

外部工具 FP16[-1,1] 输入先转 FP32[0,1]。每个原生输入均使用 core512、halo64、stride256 和 sigma128 Gaussian 混合。每次 NN 输入都是640×640，包括512 benchmark；不缩放、不按 case 识别。核心区域的 NN 残差以 FP32 混合后加入 FP32 原生输入，最后 clamp[0,1] 并返回 FP32[-1,1]。NN 运算为 FP16，LayerNorm 统计为 FP32；NN 输出减输入在 FP16 加法之后计算，极小学习残差仍受 FP16 ULP 限制。每次调用分配新输出，不缓存图像或结果。

CPU 证明见 `identity_cpu_proof.json`：LayerNorm 前向及输入/仿射参数梯度最大误差 FP64 2.66e-15、FP32 2.86e-6；实际模型小尺寸前向恒等；33×47、512×512、513×777 的 tiler 恒等和输出所有权检查通过。100张 A7750 test 原生伪标签 PNG 的真实 SHA、RGB 尺寸和输入 SHA 已校验。没有读取验证集 GT 进行训练；这些伪标签后续也不构成新 GT 泛化结论。

探测模型目录：`/cache/independent_sr_cuda_20261013/submissions/naf_push7_identity_probe/model_dir`。探测 raw：`/cache/independent_sr_cuda_20261013/adaptation/runs/naf_push7_A7750_fidelity/step0.pth`，SHA `2081d0eed44f27c6183a5b4e420ff8d0fc10969504e28fc1423bef87c13c8830`。该目录只有模型、runner、architecture、config；teacher 路径仅作为 metadata，infer 不读取 teacher 或训练资料。

训练尚未实施。若测速合格，唯一后续方案为 test100 A7750 原生伪标签、同坐标640上下文/中心512损失，fresh Adam，L1 + .1 LPIPS + .2 gradient，无 NR reward。先实测100步耗时，再500步强停并看四例；合格后沿同分支续1000/2000。原生长图视觉与正式103/418必须单独验收，不能继承 A 的指标。
