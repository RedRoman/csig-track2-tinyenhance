# Push4：精确resume＋原始raw ZIP恢复recipe备份

本轮共享存储剩余约3.899 GB，使用新目的地 `/temp/independent_sr_cuda_20261013_artifacts/nr_push4_20261014`，旧microtune、push2、push3三个prefix全部只读。新助手 `code/push4/backup_selected_push4.py` 默认仅dry-run，只有根代理显式 `--execute` 才复制。默认256 MiB空闲余量是自己的容量选择，不是用户要求。

持久化范围为：选中新阶段5的一个不可变 `resume_step7500.pt` 或 `resume_step7750.pt`（FP32 LoRA＋全部Adam矩及精确四组RNG）、原始raw ZIP的 `recipe.json` 和 `raw_zip_gaps.bin`、原字节恢复工具、真实选中checkpoint的恢复PASS证明、原103张完整评分指标/manifest/protocol/审计等新小文件，以及可选4张原始test PNG（默认case7/16/41/83，`--preview-cases none`可关闭）。不复制冗余raw checkpoint，不复制完整103张原图；这些原始输出继续保留在/cache。

manifest/DONE必须明确写出 `full103_PNGs_persisted=false`、`raw_checkpoint_persisted=false`、`raw_exact_byte_recovery_available=true`。完整103张PNG仍逐个从/cache读取SHA，与渲染及评分逐图记录核对，作为质量证据；这不能表述为已把103张原图持久化到共享存储。

根代理最终selection `results/nr_push4_selection.json`沿用上轮扁平字段和独立full/training审核路径＋SHA，新增：

- `raw_recipe_dir`：新 `audits/push4/` 内，含原工具生成的recipe.json和raw_zip_gaps.bin。
- `raw_recipe_restore_proof`、`raw_recipe_restore_proof_sha256`：对真实选中新checkpoint执行完整原字节恢复并比SHA的PASS证明。字段沿root已验证的parent7250 probe：`status=PASS`、`CPU_only=true`、`raw_restore_complete_byte_SHA256`、`resume_sha256`、`helper_sha256`。

选中计数必须是local250/global7500/Adam1500，或local500/global7750/Adam1750。可以选择中途7500，即使run后来已完成7750。助手不会用当前run终点代替不可变选中快照。

恢复工具为root新文件 `code/push4/checkpoint_archive_recipe.py`，SHA `43b135a68047a9319e355d9967b4c1bcddcab16ff58d775a8c8f7ef261d932ce`。该工具不导入torch/不解pickle，将raw514段tensor内容映射到resume ZIP entry，并保留raw的全部ZIP头、pickle元数据、gap与尾部原字节。只要resume和gaps保持SHA，恢复出的整个raw ZIP（不只是张量数值）必须与原始raw SHA完全相同。父7250已经真实验证；新选中权重还必须有自己的恢复证明，不能沿用父probe作为新winner证明。

助手在备份前再独立按recipe流式计算整个原始raw字节SHA，不写出额外raw大文件。它也保留纯stdlib白名单checkpoint检查：514个FP32 LoRA/259474432元素、原始raw与resume全部实际tensor storage SHA相等、514份Adam FP32矩形状/actual step、四组RNG存在，并要求SHA锁定的独立CPU训练审计对所有有限值、矩和RNG给出PASS。

默认dry-run命令：

```sh
/cache/independent_sr_cuda_20261013/env/bin/python \
  /cache/independent_sr_cuda_20261013/code/push4/backup_selected_push4.py \
  --selection-json /cache/independent_sr_cuda_20261013/results/nr_push4_selection.json \
  --plan-out /cache/independent_sr_cuda_20261013/nr_push4_20261014/preparation/selected_backup_plan.json
```

最终selection、fresh calibration、报告、状态和全部writer结束后，检查scope/实际空间是否通过，再用同样参数加`--execute`。同一个plan-out从payload排除，避免递归清单。若决定不持久化4张预览，两次都加`--preview-cases none`。所有SFS文件直接写最终路径并完整重读SHA＋大小；只有实际完整SHA和大小吻合的已有文件才跳过。不chmod/copy2/rename/删除任何SFS文件。已提交的新prefix同样不可覆盖。

最终manifest/DONE在源stat/全部旧头仍吻合、实际空间余量满足后才写入。单独的post-commit独立核验仍须完整重读SFS所有文件，并可从实际SFS resume＋gaps再次流式重建raw SHA；不要声称raw副本或完整103 PNG已经存在于新prefix。

从新prefix恢复raw的命令示例（输出必须是新的/cache路径，不能覆盖已有文件）：

```sh
/cache/independent_sr_cuda_20261013/env/bin/python \
  /temp/independent_sr_cuda_20261013_artifacts/nr_push4_20261014/code/push4/checkpoint_archive_recipe.py restore \
  --directory /temp/independent_sr_cuda_20261013_artifacts/nr_push4_20261014/audits/push4/REPLACE_SELECTED_RECIPE_DIR \
  --resume /temp/independent_sr_cuda_20261013_artifacts/nr_push4_20261014/adaptation/runs/REPLACE_SELECTED_STAGE5_RUN/resume_step7750.pt \
  --output /cache/REPLACE_NEW_RESTORE_DIRECTORY/step7750.pth
```

raw恢复后工具会验证完整原字节SHA；从7500恢复时相应替换步号。原评分工具/输入、旧父权重和原训练帮助源从旧3个只读prefix取，SD2基础权重继续引用共享 `/temp/zihaohe_stage/baselines/sd21b`；不重复复制env或基础模型。

新备份助手冻结SHA256：`afc2f54bffddd1e9d03cf263881f1ba4beb6022c745fc6639b7f902e0dee6cf4`。CPU只读组件试验已从现有parent7250 resume＋gaps重新流式算出完整1,038,079,419字节raw的原SHA；不写恢复副本、不导入tensor库、不复制SFS。最终Stage5选中版仍要求自己的独立恢复证明。
