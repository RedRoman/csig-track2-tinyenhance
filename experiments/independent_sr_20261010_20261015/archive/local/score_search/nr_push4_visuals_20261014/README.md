# Push4 最终full原像素图库

最终赢家为 **A7750**（`push4_a7750_compiled_all`），与父7250、C7750、D7500保留同场比较。正式selection原字节SHA为 `b3148b21f466854f9fe8f81c7eca1348da5bf793141799573ff3be8b20e25c3f`。三个新候选均已完成正式val3+test100，逐一从103行/418项指标重算NR、FR与S并与独立full审计的选择记录一致。

`index.html`展示4版×4固定例（7/16/41/83），共16张原始PNG。父版4张从旧Push3图库硬链接复用；新候选12张逐一对正式最终manifest、远端实际PNG SHA、本地原图SHA、RGB/原生分辨率和逐图指标记录。没有改写原图或旧目录。

概览严格为1/8 Lanczos缩略。ROI沿用上一轮固定整数坐标、1:1原像素，不缩放、不锐化、不润色。`final_visual_audit.json`对16块ROI与原图直接crop逐像素相等、概览与声明操作相等、所有原图SHA不变及HTML链接逐项PASS，并绑定真实最终selection。`final_candidate_binding_cpu_audit.json`额外核验三个候选的完整分数/manifest/选择向量，以及远端实际训练、渲染、runner、model config和HYPIR源码SHA。

四例肉眼差异较小；选中A的3张验证图FR均下降，固定验证负例另见相邻`nr_push4_val_visuals_20261014/index.html`。D保真较稳，完整S没有超过父版，未选中。示例不替代全量评分，测试图无GT，不能据纹理较锐宣称细节真实。

图库源码SHA：`17ba6a8e59740e73df386198b83d18838e607c707ecaae4b371bc9c33ded7634`。`source_history/`保留本轮扩展到三候选前的源码及diff，旧Push3源码和原图保持不变。两份取图watcher均已exit0；最终明确执行无`--allow-progress`fetch、`select`、`build`与`audit --final`。

服务器只保存此图库的小元数据证明到`audits/push4/visuals/`，不上传本地图库/ROI/原始PNG。SFS备份的4张选中预览与完整103PNG保留范围由最终备份清单单独说明，不把本地图库误称为全部103张已持久化。
