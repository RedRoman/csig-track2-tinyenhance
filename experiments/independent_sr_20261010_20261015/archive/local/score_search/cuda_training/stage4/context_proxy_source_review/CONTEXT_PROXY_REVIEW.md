# 全图上下文代理：仅供后续阶段的源码分析

本记录没有启动GPU计算、修改评分工具或改变正在运行的v4训练。依据是服务器当前安装的pyiqa 0.1.10源文件与原始`csig_metrics.py`，文件已下载到本目录并记录SHA256。以下区分源码事实与尚未实测的推断。

源码事实：原评分的`load_img`保持native RGB FP32尺寸；`measure`直接把全图交给四个NR指标，没有统一resize或crop。`InferenceModel.forward`也把张量直接交给相应网络。MUSIQ默认KONIQ模型的`longer_side_lengths=[224,384]`，`max_seq_len_from_original_res=-1`，因此评测同时包含长边224/384的整图缩放和全部原分辨率32×32 patches；多尺度位置按整张图的网格hash。CLIPIQA仅做CLIP mean/std，然后以`pos_embedding=False`将native图交给RN50，不是先裁成224×224。相关位置为`csig_metrics.py:91/101`、`inference_model.py:62`、`musiq_arch.py:323/373`、`multiscale_trans_util.py:147`、`clipiqa_arch.py:147`。

当前512训练会把局部ROI本身当成整张图。其224/384缩放内容、空间hash及native token数量都与4K正式评测不同。正式100图全部为4096×3072或转置，MUSIQ序列加CLS后为12482 tokens；512×512训练为450 tokens。这是正式MUSIQ在提高训练权重后仍下降的一个合理解释，当前证据不足以断言它是唯一原因。CLIP也有全图语义与局部纹理的类似差异。

若当前v4 pilot趋于平台，建议先做一个新的小规模上下文分支，不改v4源。优先使用当前最佳full PNG为固定、detach的完整背景P，仍只生成与teacher共用同一latent的512 ROI。不要直接把student ROI覆盖P，因为本地crop生成与native全图生成在VAE上下文、噪声和wavelet边界上并不完全相同，容易产生人工接缝。更稳妥的训练代理为：

`context = clamp(P + scatter(mask * (student_ROI - teacher_ROI)), 0, 1)`

teacher必须是生成P的同一最佳版本，且FP32精确恢复。开始时student等于teacher，残差约零，context就等于正式父PNG；这避免了初始局部重绘与全图背景的固有不一致。mask可以在32或64像素边缘平滑降至零，内部保持1。它是训练损失代理的一部分，不能用于正式输出或修改正式指标。仍需记录mask、位置、P文件SHA及tensor范围，固定同一crop/RNG协议做对照。

MUSIQ/CLIP先看此context长边1024的保长宽比缩放，MANIQA继续在完整可导的512 ROI上计算，保真项也继续约束512 student与teacher。1024×768的MUSIQ序列为962 tokens，保留全图构图而显存负担远低于4K。若这一训练代理带来真实完整NR/S改善，再测试1536长边；任何缩小后的loss都只能称训练代理，最后仍须用原始native4K和原工具正式测量。

不建议第一步就对4K context做完整MUSIQ反向。当前实现每层显式构建6头N×N attention矩阵；N=12482时单个FP32矩阵约3.48GiB，14层仅这一类保存量就约48.8GiB，尚不含其余激活。HYPIR现有训练峰值约29.95GB，二者联算预计显存紧张；这里是源码量级推断，尚未进行实测。CLIP RN50还会增加全图卷积激活。必要时可以逐指标计算输入VJP、立即释放指标图，再将加权ROI梯度回传student，但这应作为单独新实现验证一阶梯度等价和loss日志语义，不能临时改动正在运行的源码。

主要限制有五个。第一，固定背景只让critic在一个局部ROI求导，不等同于真实全图模型同时改变所有区域，背景随训练变旧，需要短阶段和独立full确认。第二，smooth mask与裁剪本身会影响边界梯度，critic可能优先修补代理接缝；残差式零起点可以缓解但不能消除。第三，ROI仅占4K图面积1/48，下采样又让512区域缩成约128×128，global reward梯度可能较弱；不能直接把reward乘48当成已校准做法，应先记录各项实际输入梯度范数并限制放大。第四，全图MANIQA随机224 crop常常不覆盖ROI，导致该项对可导ROI梯度为零或稀疏；保持MANIQA局部512分支能避免这个问题。第五，512 crop latent噪声、wavelet和native tiled推理仍不完全相同，因此上下文代理只减少构图差异，不会自动消除训练/正式输出差距。

建议后续最小实验为同一新best父权重/RNG下两分支各250步：旧局部loss对照与新增全图context MUSIQ/CLIP代理，reward系数先保持可比并记录实际梯度；固定20+val3筛选后对领先者完整100+3测量。当前v4三分支先按已封存750步计划完成，不因这份分析改变源或追加未验证的训练目标。
