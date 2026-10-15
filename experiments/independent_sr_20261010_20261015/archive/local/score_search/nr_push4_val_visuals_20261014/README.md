# Push4 最终固定验证对照

比较GT、父7250、C7750、最终赢家A7750与D7500的验证case1，共5张原始PNG。固定沿用上一轮ROI `[2304,2688,2944,3200]`，不以本轮GT或候选差异重新选病例/区域。GT与父图硬链接复用旧Push3原图，新候选仅各下载1张正式full PNG，逐SHA验证。

所有4个模型的正式full103图/418项评分均完成后运行`diagnostic.py --additional-label push4_d7500_compiled_all --additional-checkpoint-sha256 feb55a9e22e466e6f45f3b15d6cd38b67a679d761fc37e093b77687fafda3b50 --final`。`audit.json`逐像素核对5块1:1 ROI，并确认全部最终manifest和原始PNG SHA。总览另为1/8 Lanczos缩略，没有锐化、滤波或润色。

本页保留A/C三张验证FR均下降的事实；D在case1提高、case2/3近似持平，完整S仍未胜出。这里只报告逐验证图FR，不由单例推断测试均分。`final_selection.json`保留真实选择原字节，`final_selection_binding.json`将赢家A的raw SHA、val1原图SHA和最终主图库/负例审计绑定。

诊断源码SHA：`1c45627d739e81d0904c369ad17a0c4f8f94dfe952ae62d6f29ada5fb670436b`。服务器只上传源码、JSON、说明、HTML等小证明到`audits/push4/visuals/val_diagnostic/`，所有原图与图库PNG留在本地新目录；旧图库与全部旧备份保持只读。
