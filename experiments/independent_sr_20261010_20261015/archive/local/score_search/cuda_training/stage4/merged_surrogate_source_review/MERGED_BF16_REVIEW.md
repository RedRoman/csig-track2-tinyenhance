# 部署融合与训练forward的精度差异：后续阶段可行性

本记录只读取当前安装的PEFT、Diffusers、Torch与HYPIR源代码，并在CPU上读取已封存6500 checkpoint的形状；没有构建模型、运行GPU、改变当前v4训练或正式Runner/指标。源文件副本与SHA位于本目录，CPU形状证据为`CPU_shape_cost.json`。

当前部署路径在`runner.py:80`调用`G.fuse_lora(scale1,safe_fusing=True,adapter_names=['default'])`再`unload_lora()`，之后才compile。HYPIR `base.py:32`把base设为BF16，`sd2.py:20`先构建BF16 UNet、add_adapter，然后load_state_dict。Diffusers `peft.py:543`经PEFT `mapping.py:47`注入，PEFT `tuners_utils.py:1070`把新adapter移到base的device/dtype；该inject路径未调用另一个PeftModel包装器的自动FP32 upcast。因此源码链支持当前推理A/B载入到BF16、再在CUDA上融合；后续实现仍应记录每个实际模块的运行时dtype，不能仅凭模型文档假定。

当前训练则在初始化后把514份LoRA显式转FP32，并从resume精确恢复FP32主参数，forward位于BF16 autocast中。PEFT Linear的未融合forward（`layer.py:744`）是`base(x) + B(A(x))*scale`，中间A(x)、B(...)、相加与cast有各自舍入；融合后变成`linear(x, round_bf16(base + delta_bf16))`。有限精度下两条路径不满足分配律等价。此前PNG STE只处理输出量化，没有覆盖A/B载入量化、delta计算或融合后权重的量化。

理论上可以另建训练阶段，保留原514份FP32主参数与Adam状态，仅forward使用可微的融合权重。需要精确复现当前PEFT CUDA运算序列，而不是在FP32算完B@A后只cast一次：

1. A/B主参数先分别cast到BF16，保留指向FP32主参数的autograd边。
2. Linear按`layer.py:710`做B@A、fan-in/out处理及scale；1×1 Conv按`layer.py:1272`做squeeze后matmul；其他Conv按`conv_fn(A.transpose(0,1),B)`再transpose和scale。当前alpha=rank使scale为1，DoRA/RSLoRA/bias/dropout均未启用。不能把所有Conv简单换成二维matmul并假定bitwise相同。
3. 对应safe_merge（`layer.py:629`与1162）先取原BF16 base，再加已cast为base dtype的delta，结果落回BF16；base必须detach，不用Parameter.data写入，不反复merge/unmerge累积误差。融合运算应在明确的autocast策略下执行，复现CUDA原op与dtype；CPU PEFT的BF16→FP32特例不能拿来作为CUDA数值等价证明。
4. 把这些非leaf Tensor传给无LoRA重复分支的plain UNet skeleton，例如`torch.func.functional_call`替换对应权重，其他base参数与buffers冻结。不能再执行一次原LoRA residual；也不能把merged Tensor重新包装成nn.Parameter，因为那会切断到原A/B主参数的梯度。参数映射须处理原`.base_layer.weight`与plain权重路径、bias和共享权重，不应只靠未经验证的字符串替换。

常规`.to(BF16)`反向会把梯度cast回主参数dtype，相当于对离散舍入使用straight-through近似；这不是BF16离散程序的真实数学导数。可做到“更忠实的融合forward＋代理backward”，但不应称精确部署梯度。FP32主参数微小更新也可能尚未跨过BF16格点，实际融合权重暂时不变；需要记录融合参数变化比例及正式PNG/完整分数，不能只看训练reward。

CPU实际统计为257个adapter模块：192 Linear、14个1×1 Conv和51个其他Conv。对应dense融合权重共837,602,560元素，一套BF16约1.675GB，原LoRA FP32主参数约1.038GB。每步重新构造dense delta及其反向，还会增加BF16 A/B副本、merged/delta存储及dense权重梯度，不能按“只训练LoRA”假定显存或速度不变。仅秩256的dense delta生成约需0.429TFLOP/步，实际时间还包括257层launch、Conv、dense权重反向和访存；这里是形状量级估计，尚未实测。当前512训练峰值约29.95GB，A100 80GB可能有容纳原型的余量，但必须先做单步显存/耗时测量。

最大的实现陷阱是gradient checkpointing。`functional_call`只在调用期间临时替换参数；如果UNet内部checkpoint在反向阶段重算时原functional_call上下文已经退出，重算可能读回原base权重，从而得到错误forward或丢失A/B梯度。不能简单在当前已开启内部checkpoint的UNet外套一次functional_call。后续原型须让替换操作本身处于每次重算的函数内部，或采用逐层局部融合wrapper，并明确选择/验证checkpoint方式。还需核对Torch functional_call文档中的in-place、parametrization、tie_weights与strict映射约束（本目录`functional_call.py:12`）。

若后续尝试，先以同一checkpoint、固定相同latent和conditioning比较每层实际merged tensor、eager fused UNet输出和代理forward；再核514份FP32主参数的真实非零有限梯度及Adam续接。之后才做相同crop/RNG的短250步对照、固定pilot和native完整100+3评分。即使融合权重逐项bitwise相同，eager训练与compiled部署仍可能存在kernel舍入差异，不能直接宣称整个Runner已bitwise等价。当前v4与本轮full评分保持原封存实现。
