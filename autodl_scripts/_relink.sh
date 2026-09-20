#!/bin/bash
# AutoDL 容器重建后跑这个，恢复 /cache/zihaohe 软链（脚本里硬编码了该路径）
mkdir -p /cache
[ -e /cache/zihaohe ] || ln -s /root/autodl-tmp/zihaohe /cache/zihaohe
ls -ld /cache/zihaohe
