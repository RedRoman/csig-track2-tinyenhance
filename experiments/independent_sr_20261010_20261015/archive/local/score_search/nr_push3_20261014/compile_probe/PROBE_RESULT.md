# reduce-overhead 编译探针（有界，未完成测速）

GPU3 独占探针在 155.27 秒总预算后停止自有进程组，GPU3 的随后 nvidia-smi compute-apps 查询为空。GPU0/1/2 未使用。

6500 M1.5 原权重 SHA256 `cfcc3013fa87aa721f346d6cc285e70b6d6ff2c88315ce9f1465b07ffe020e68`；原 runner `396e29afc3b60e077a8ef200cb4c382b33abac4fb60ab37dd9a873eb369704ae`；新 runner `74f138aa24be01a7075a5a45b469a62ae70c9fe2dedc10c705a41a784e2c2d12`。仅将 UNet/VAE 三处 torch.compile 的 mode 从 default 改为 reduce-overhead，并同步协议文本。旧 runner 保持不变。

Torch2.8.0+cu128 的本机文档说明 reduce-overhead 使用 CUDA Graph 降低 Python 开销，要求 CUDA-only 图且不修改输入，可能增加缓存内存；另保留 cudagraph_mark_step_begin 关于输出生命周期的文档。见 torch_local_docs.json。

新旧各自进程，恢复同一组 Python/NumPy/TorchCPU/CUDA RNG 后，连续 3 次 512x512 推理全部 finite，输出逐 bit 相同（max_abs=0），每次结束的四类 RNG 也完全相同。此证明只覆盖这 3 次 512 调用；没有证明 native4K 输出和完整质量评分相同。

原 runner fresh-process、warm30、n200：CUDA event p10=46.15638962 ms，median=46.61299324 ms。原测速工具 SHA256 `3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995` 未修改。

新 runner 的 fresh-process benchmark 剩余时间仅 21.93 秒，编译未能在总预算内完成，未得到延迟数值。该候选标记为 incomplete/failed，不能宣称加速或工具预测分数提高。保留探针目录供核查，但不作为已通过的正式 submission。
