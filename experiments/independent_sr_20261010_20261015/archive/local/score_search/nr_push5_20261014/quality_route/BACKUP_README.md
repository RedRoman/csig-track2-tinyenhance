# Push5 选中学生备份

`backup_selected_push5.py`为131行Python标准库助手，root执行；默认dry-run，不会复制SFS。正式选择文件须为`results/nr_push5_selection.json`，状态`selected_new_student_full103_paired_prediction_ge70`，并绑定实际训练/full审计和新鲜配对结果的路径与SHA。助手从paired的`newly_trained_student.original_tool_prediction`核实至少70，不使用随包G4参考复现的分数冒充新训练。

只持久化新学生完整可携带model_dir、匹配immutable resume_step250.pt、原G4初始化model.pth、4张学生原生PNG（7/16/41/83）及必要的新源码/配置/marker/train.jsonl、三个质量目录的小元数据、独立审计、六份final配对JSON、中文报告与selection。三个质量目录为G4参考新full、旧A7750 full和新学生250 full。

不另存一份step250.pth：deployment的model.pth与它逐字节相同，可原样拷回恢复raw。完整103张学生PNG和100张A7750训练伪标签仍留/cache，未全复制到SFS；继续训练需要这些教师PNG或按原封存模型重新生成。该prefix不是环境、输入与全部旧模型的独立副本。

旧四prefix只读取固定manifest/DONE头，不删除、覆盖或重写。Push4的固定receipt SHA为`ea9f3a3d…9ad14f`，对应manifest `3e1c41dc…cd35a`、DONE `ac09208e…e722f`。新prefix是`/temp/independent_sr_cuda_20261013_artifacts/nr_push5_20261014`。

默认自选空闲余量128MiB，不是用户要求。实际文件逐SHA/size编制清单，先检查空间；只复用实际大小和SHA均匹配的partial文件；新文件直接流式写最终路径并完整SHA读回。提交前再检查所有源stat与旧头不变，最后写manifest和DONE。拒绝新prefix中的symlink或非清单文件。备份日志与计划放R外，示例：

```sh
/cache/independent_sr_cuda_20261013/env/bin/python /cache/independent_sr_cuda_20261013/code/push5/backup_selected_push5.py > /cache/push5_backup_dry_run.log 2>&1
/cache/independent_sr_cuda_20261013/env/bin/python /cache/independent_sr_cuda_20261013/code/push5/backup_selected_push5.py --execute > /cache/nr_push5_backup_execute.log 2>&1
```

计划默认为`/cache/push5_selected_backup_plan.json`。root应先结束并冻结所有被收集的新报告、审计、配对和源码写入，再dry-run/execute。只有实际DONE和执行exit0成立后，另做提交后的独立核验与新receipt，不能改归档内报告/manifest/DONE制造自引用。
