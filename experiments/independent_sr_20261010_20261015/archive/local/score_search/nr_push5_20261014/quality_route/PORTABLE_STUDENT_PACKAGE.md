# 新训练学生提交目录

实际250步学生目录：`/cache/independent_sr_cuda_20261013/submissions/pix_push5_g4_nr250_s250/model_dir`。该模型由随包G4参考权重初始化，使用A7750原生PNG伪标签真实训练250步；不是把参考复现成绩改名为新训练成绩。

ZIP只保留一个顶层`model_dir/`，内含这6个普通文件：

```
model_dir/
  runner.py
  reference_runner.py
  pix_model.py
  model.pth
  student_binding.json
  config.yaml
```

合计12,705,240字节。`model.pth`为真实新step250，SHA256 `8c95e115e12219bca8bea2b2d04e0a01552f072de56619fab878ae61a9643237`；metadata wrapper SHA256 `c5f59b7e42f3050d0bc62b1fccdcc38a6e01c91edbd54d3c03061fe5f2129abd`。原G4 runner与pix源码分别为`6222ead9…d8c1e`、`5b1ba759…50f5f`，没有改推理数学或添加输出后处理。

不打包`__pycache__`、`.pyc`、`_work`、编译缓存、训练数据、教师PNG、训练日志或resume。硬链接在ZIP里保存实际文件字节，不需要原服务器的链接目标。`student_binding.json`内的训练路径只作来源记录；运行时只读取此目录内的文件，导入依赖为Torch和Python标准库，不读取HYPIR/SD权重或其他/cache资源。保留的35字节config.yaml与参考件一致，原Runner实际从本目录model.pth加载。

原G4推理方式保留：512输入使用原自选编译/graph路径，原尺寸输入使用原eager并保留OOM分块回退。512测速不是4K整图延迟。训练指标网络不属于提交运行时依赖。
