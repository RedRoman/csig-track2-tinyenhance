# ModelArts 第三方源码补丁

这两个补丁保存了 30674 上第三方仓库克隆的未提交修改；仓库本身不收录第三方源码。

| 补丁 | 上游提交 | 用途 |
| --- | --- | --- |
| `VOSR-local.patch` | `cswry/VOSR` `d72c64d` | 优先从本地 torch hub 加载 DINOv2 |
| `OSEDiff-local.patch` | `cswry/OSEDiff` `cdd3b7b` | 使用 ModelArts 本地 BERT 目录 |

在对应上游提交的仓库根目录运行 `git apply /path/to/patches/<补丁文件>`。
OSEDiff 补丁中的 `/cache/zihaohe/...` 是 30674 上的绝对路径，换机器使用前需调整。
