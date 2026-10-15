# Push3 仅最终选中版本备份

新助手部署路径为 `/cache/independent_sr_cuda_20261013/code/push3/backup_selected_push3.py`，新共享存储目的地固定为 `/temp/independent_sr_cuda_20261013_artifacts/nr_push3_20261014`。它不会写入旧备份。默认仅检查和输出计划；只有显式 `--execute` 才复制数据。默认3 GiB空闲余量是本次助手自己的选择，不是用户要求，可通过 `--reserve-gib` 调整。

本地目录中的代码、模板和静态审计属于准备文件；尚未有最终selection时不生成虚假的PASS计划，不复制任何SFS大文件。最终根代理须在全部必要完整评测及CPU审计完成后，写入 `results/nr_push3_selection.json`。该JSON应包含：

- `status`: `selected_after_independent_full_audit`
- `label`, `NR`, `FR`, `S_proxy`: 选中版本完整100张test＋3张validation的真实结果。
- `local_step`, `global_step`, `optimizer_step`: 例如250/6750/750，或500/7000/1000，或750/7250/1250。当前训练可以已经运行到更后面的步数。
- `checkpoint`, `checkpoint_sha256`: 选中的原始 `step{global}.pth`。
- `resume`, `resume_sha256`: 对应不可变 `resume_step{global}.pt`，拒绝mutable `resume_latest.pt`。
- `config`, `model_dir`, `quality_dir`: 对应训练配置、实际全量渲染采用的推理模型目录和全量评测目录。
- `full_audit`, `full_audit_sha256`: 原工具全量103行/418个标量的独立PASS审计文件绝对路径和SHA。
- `training_audit`, `training_audit_sha256`: `audits/push3/training_stage4/milestone250_cpu_audit.json`、500或750版本之一，必须明确包含选中branch/milestone的PASS记录。
- 若使用其他新renderer源文件，额外指定 `renderer_source`；省略则使用原 `code/render_cuda_submission.py`。其实际SHA必须等于渲染协议内的记录。

助手按已存在的真实完成证据选中6750/7000/7250，不要求选中步等于当前run状态或训练终点。它核验保存标记与不可变resume的源大小/mtime/inode，随后通过纯标准库ZIP＋惰性白名单pickle读取LoRA/Adam/RNG元数据；逐一核验全部514份原始权重与resume中FP32 LoRA的实际storage SHA一致，Adam矩的FP32/形状与每份step正确，四组RNG存在。有限值、全部矩、实际迁移及4组RNG可恢复性还必须由SHA锁定的独立CPU训练审计PASS证明。备份不会导入torch或NumPy，也不会创建CUDA环境。

全量质量核验会读取103张实际PNG的SHA，与render_manifest及逐图评分记录逐一对应，核验418标量并重新计算各均值、NR/FR/S。推理目录必须与全量渲染协议一致；协议记录的配置和源文件SHA仍须吻合，因此只有根代理最终选中的推理/graph变体能进入备份。共享SD2权重和原评分网络只核验原有字节与旧备份引用，不复制。

复制范围严格限制为：1份选中raw权重、1份匹配的不可变resume、1份选中完整103张PNG、选中run小日志/配置/证明、选中推理目录小配置/代码、`code/push3`、`audits/push3`、新增Push3报告/结果/状态/校准小文件、v4源和3组配置及本次准备文档。其他候选/里程碑的大文件、其他全图输出、旧全部code/reports、env和SD2基础权重不复制，仍留/cache。

推荐先执行dry-run并保存计划：

```sh
/cache/independent_sr_cuda_20261013/env/bin/python \
  /cache/independent_sr_cuda_20261013/code/push3/backup_selected_push3.py \
  --selection-json /cache/independent_sr_cuda_20261013/results/nr_push3_selection.json \
  --plan-out /cache/independent_sr_cuda_20261013/nr_push3_20261014/preparation/selected_backup_plan.json
```

核对实际`space_sufficient`及范围后，使用完全相同参数再加`--execute`。生成计划的路径始终通过`--plan-out`明确传入，助手会从其自身payload中排除该文件，避免递归清单。所有要归档的报告/审计/状态须在最终执行前写好并停止修改。

新目的地上的每个文件均直接写最终路径，再读取完整SFS字节核验SHA与大小。部分复制只会在实际完整SHA和大小都匹配时跳过；不能仅凭已有manifest或文件长度信任大文件。写入不调用chmod、copy2、rename或删除。空间核验按尚需增加的实际字节计算，并在每份文件写入前再次检查。已有错误或不完整的新prefix文件可以在同一路径重新流式写入；旧prefix完全只读。

最终所有被选源文件/已核验外部源的stat保持不变，旧两份backup head仍吻合，空闲余量满足后，才直接写 `manifest.json`，读回校验，再写包含manifest SHA的 `DONE`。如已有完整提交，只接受同一selection及同一文件清单，完整重读SHA通过后返回幂等成功，不覆盖已完成提交。

归档后的inode/mtime无需等于/cache源；源inode只证明选中不可变里程碑未被替换，SFS读回证明以字节SHA为准。

旧备份引用固定为：

|备份|manifest SHA|完成标记|
|---|---|---|
|nr_push2_20261014|ef284432a338b0a4574f52be645f4a40571ef94e775e1e39a2c0d1cfc1d13e8d|DONE|
|nr_microtune_20261014|99f461e6e95045cf0af053dc3b24bde74cd3531715225d0ccf95eef2d2dbd0d4|DONE.json|

两组实际DONE与manifest已在本次准备时只读核验，助手还会在准备及提交前核验同一SHA。旧payload不在此处重新复制；恢复时需要以上两个旧prefix与共享SD2路径共同提供完整依赖。

实际运行使用的CLI配置均部署于 `code/cuda_training/` 根目录，助手会强制包含 `config_stage4_A_continue6250.json`、`config_stage4_B_reanchor6500.json`、`config_stage4_C_nrstrong.json`、`stage4_preparation_static_audit.json` 和 `STAGE_V4_README.md` 五个实际文件，缺失即终止。新助手当前SHA256：`60fbeae585b4fcb6475f728477166f868880d4dd4707d8ce5695702a718156b7`。原v4训练源保持原字节。
