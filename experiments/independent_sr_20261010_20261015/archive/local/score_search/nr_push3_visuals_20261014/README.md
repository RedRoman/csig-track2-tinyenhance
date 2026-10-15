# 第三轮 full 原图比较

`gallery.py`只在CPU上处理已生成的full103原图。父6500的case7/16/41/83已从上一轮本地原PNG建立硬链接，并逐文件核对最终full manifest的SHA和尺寸；初始化没有下载任何新候选PNG。

收到明确full候选标签、远端full目录与checkpoint SHA后使用：

```text
python3 gallery.py fetch --label FULL_LABEL --directory /cache/independent_sr_cuda_20261013/nr_push3_full_evaluations/FULL_LABEL --checkpoint-sha256 EXACT_SHA --name "New best"
python3 gallery.py build
python3 gallery.py audit --final
```

仅在需要预览已经committed的full进度时加`--allow-progress`；pilot目录始终拒绝。progress预览之后必须对同一候选再fetch一次最终full manifest并通过`audit --final`，才作为最终交付。每个新候选最多下载四张原PNG，图库支持父版加一或两个候选。不下载整套full PNG或权重。

ROI沿用上一轮相同整数坐标：case7=(1856,1408,2496,1856)，case16=(1280,1216,1920,1728)，case41=(768,1664,1408,2176)，case83=(1216,1920,1856,2432)，坐标按[left,top,right,bottom)且源图左上角为原点。ROI仅裁剪和原样拼接，不缩放、不锐化。全图总览单独标明为原图1/8、Pillow Lanczos缩略。审核会逐像素比较每块ROI及总览预期采样结果，并检查所有HTML链接和源SHA保持不变。

完整NR/FR/S和分项均由完整100测试+3验证的103行/418项指标独立重算，只在result完整时展示；四张示例仅列各自full序列的逐图指标，不推断全量均值。观察文字写入`visual_observations_zh.txt`后重新build和audit；应按实际原图记录纹理、边缘和平滑变化，不能仅因指标上升便声称细节真实恢复。

最终备份前，将gallery.py、README、sources.json、preview_manifest.json、selected_per_image_metrics.json、final_visual_audit.json、index.html、visual_observations_zh.txt及provenance内小型JSON/JSONL上传服务器`audits/push3/visuals/`。新best native PNG本体由selected-only正式输出备份保存；本地图库保留四张下载原图和父版硬链接。
