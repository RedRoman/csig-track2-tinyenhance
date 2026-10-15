> 本次汇总的当前现场状态以[总交接](README.md)和[资源快照](archive/server_cuda30416/LIVE_RESOURCE_SNAPSHOT.json)为准；本页中历史阶段的在线性、缺失资产与旧选择仍按来源时间解释。

# PixG5i → 固定 HYPIR 教师蒸馏实验草稿（2026-10-15）

本文件记录已经发生的实验，机器清单为 `pixg5i_versions.json`。共14个版本：10个完成103张/418项完整质量评分，4个未完整评分；仅原模型与mild300/medium300做过一轮同卡五进程正式测速。所有新蒸馏候选均未采用为主模型，原包与HYPIR教师指针保留。最新coverage v4在200步未通过结构视觉门槛，停止继续训练；完整评分用于保留失败对照，没有再做正式测速。

## 位置、端口与模型身份

- 本地实验根目录：`/Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/pixg5i_hypir_distill_20261015`
- 远端根目录：`/cache/independent_sr_cuda_20261013`；实验目录：`/cache/independent_sr_cuda_20261013/experiments/pixg5i_hypir_distill_20261015`。
- 完整评分根目录：`/cache/independent_sr_cuda_20261013/nr_pixg5i_distill_full_evaluations`。
- 本轮SSH：`huawei-modelarts-31514:30416`。这是当时会话记录，重新使用前须确认实例；历史/cache发生过重置，不能凭旧路径认定文件仍在。
- 这次整理只读取现有小metadata/config/计时文件并写本地两份文档，没有复制pth、resume、大图或整套数据，没有新实验、远端写入或GitHub操作。

原包为 `赛题二_pixG5i_g_b4e0686c.zip`，SHA256 `f8e154da5ea9cb2fe165a316f824003459661945e6bbe22efd025d8592d573d6`。原权重15,173,878 B，SHA256 `f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487`。原包PixUNet2架构：112个FP32 state张量、3,786,764参数，cfg为 `chs=[32,64,128], nbs=[1,2,5], norm=none, tdim=128, ps=2, t_fixed=999, arch_ver=2, nbs_dec=[1,2]`。原包保存的reference_step30000是历史元数据，不是此次新增训练步数。

**PixG5i不是G4参考模型。** 原Runner源码虽相同，模型源码与权重不同。本轮真正G4校准权重SHA256为 `283cb7d051a4686e558057b8fb0dcd927b47a8cb7474101fab4162c457c2bb83`，不能拿原PixG5i代替。

## 教师、数据及原推理

固定教师为最新HYPIR B（lastres LR5e−6 s7850，TRT UNet/VAE），raw SHA256 `23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c`，CM SHA256 `d7e03918355aa9cb459c1d37f8ffafcee89856bb8b23cdf0346b5e440b3046cb`。教师manifest `/cache/independent_sr_cuda_20261013/nr_push7_recovered_full_evaluations/push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae_all/render_manifest.json`，SHA256 `f1a2004d3e4fa56272bc00183a84960510dcc6066bb98b0d29930d04a7642fd6`；输出在同目录 `render/test`，protocol文件SHA256 `e83eb1eb0c0e04b6f2bd9ce7ac6c07d2662bc7173796792b0774b650d8802f11`。

训练只读test100冻结原尺寸教师PNG。输入test100也是完整NR评价使用的同100例，因此这是本轮数据适配，不能当成未见测试集泛化成绩。val图像及GT均不进训练；val3 GT只在最终FR评分时使用。JSON的datasets保留全部100输入/100教师记录的路径、尺寸与SHA，phase provenance绑定catalog原文件。

所有阶段保持原包网络及部署数学不变。adapter只提供身份metadata并在构造前固定Python、NumPy、TorchCPU、CUDA seed231，infer纯delegate。原FP32 fold→FP16/channels_last；512走原compile/CUDAgraph，大图走原native eager，仅原OOM fallback用tile1024/overlap128。推理不读教师、不混教师图，不按评分case切换模型。

## 训练设置与血缘

共同：batch1；全部112 FP32 master；FP16 autocast forward；FP32 loss/LPIPS；GradScaler初始1024。Adam betas=(.9,.999)、eps1e−8、weight_decay0、foreach=False；梯度范数clip1。cudnn_benchmark、cudnn/matmul TF32均False。输入RGB uint8/255→half()*2−1，再按原unfolded AMP频率实现计算。各阶段新建Adam0和phase RNG；同phase续训恢复immutable resume中的Adam、scaler及四RNG。

native1024 context、中心512损失、halo256，context起点对齐8。mild/medium/v2/v3逐步有放回随机选case及合法起点。RGB01 raw L1和一阶差损失；只有LPIPS输入clamp01→[-1,1]、normalize=False。梯度为dx、dy L1的平均。MS项为avg_pool2d因子2/4的RGB L1及粗网格一阶差平均，不按尺度归一化物理梯度。所有NR reward=0、无GT、无original-output anchor。

| 阶段 | 实际/计划新步数 | 初始与处理 | LR | L1 / LPIPS / grad / MS RGB / MS grad | seed |
|---|---:|---|---|---|---:|
| mild | 600/600 | 原f378全112 | 5e−6→1e−6 cosine | 1 / .05 / .1 / 0 / 0 | 20261015 |
| medium | 600/600 | 原f378全112 | 5e−6→1e−6 cosine | 1 / .1 / .3 / 0 / 0 | 20261015 |
| structure v2 | 1200/1200 | medium600全112 | 2e−5→2e−6 cosine | 1 / .05 / .3 / .5 / .3 | 20261016 |
| headreset v3 | 600/600 | v2 local600：outc两张量3468元素清0，body110精确继承 | body1e−5→2e−6；head5e−5→1e−5 cosine | 1 / .005 / .3 / .5 / .3 | 20261017 |
| coverage v4 | **200/2000，门槛失败停止** | headreset600全112继承，无新reset | body max1e−4/final2e−5；head max2e−4/final4e−5 | 1 / .005 / .3 / .5 / .3 | 20261017 |

单组cosine使用(step−1)/(T−1)；headreset同样按T=600的两组cosine。coverage前50步由最大LR的10%线性升至100%，保持至1000，再cosine至2000终点；这里只实际训练200步，没有实际执行后续计划。

coverage固定2000行计划：每100步100case无放回；完整计划中每case访问5×4起点分层各一次，仍对齐8；plan SHA256 `5466f898c854b8337d16879a867b04b14b631959df18c6baca817c9eaafe0d28`，实际next_index=200。这是case/起点覆盖，**不是全像素覆盖**，外侧256边缘仍不在中心损失。

血缘为原包→mild/medium；medium600→structure v2；**structure v2 local600（不是local1200）**→headreset；headreset600→coverage。structure local1200保留为另一个终点。headreset首步body梯度/更新0、head两张量更新是预期，后续全112可更新。

local步数和Adam从0重新计；累计新增蒸馏步数：mild/medium=N，structure=600+N，headreset=1200+N，coverage=1800+N；不加原包30000。JSON逐版本保存phase/累计计数、CM及raw/resume SHA；不是用文件名猜测。

## 完整质量和正式工具估分

原renderer/scorer固定val1..3→test1..100，native103图/418项有限标量，MANIQA100 crops。render构造前seed231、连续调用、无warmup；score每split seed1234一次，连续RNG。CPU审计绑定actual PNG/input/GT SHA、顺序、原源码/协议、原公式及published result。终点四RNG相同只代表保存的终点，不冒称逐call相同。

NR=mean_test(CLIPIQA)+mean_test(MANIQA)+mean_test(MUSIQ)/100+max(0,(10−mean_test(NIQE))/10)，先求test100均值再NIQE clamp。FR=(1−mean_val(LPIPS))+(1−mean_val(DISTS))，只用val3。**S_quality=NR+FR，不是含速度的工具总分。**

| 版本 | local / 累计新步数 | NR | FR | S_quality | 512 event p10 ms | 原工具估分 |
|---|---:|---:|---:|---:|---:|---:|
| original | 0 / 0 | 2.827770143572 | 1.279535472393 | 4.107305615965 | 0.720896005630 | 153.369110742274 |
| mild_s100 | 100 / 100 | 2.628310172306 | 1.359197497368 | 3.987507669674 | null | null |
| mild_s300 | 300 / 300 | 2.358091225082 | 1.373944004377 | 3.732035229458 | 0.719871997833 | 118.264390609125 |
| mild_s600 | 600 / 600 | 2.262832116063 | 1.379095276197 | 3.641927392259 | null | null |
| medium_s100 | 100 / 100 | 2.637747104685 | 1.358900030454 | 3.996647135139 | null | null |
| medium_s300 | 300 / 300 | 2.406142042267 | 1.373535553614 | 3.779677595882 | 0.722944021225 | 122.164436623278 |
| medium_s600 | 600 / 600 | 2.318599592824 | 1.378493070602 | 3.697092663426 | null | null |
| structure_v2_s300 | 300 / 900 | null | null | null | null | null |
| structure_v2_s600 | 600 / 1200 | null | null | null | null | null |
| structure_v2_s1200 | 1200 / 1800 | null | null | null | null | null |
| headreset_v3_s100 | 100 / 1300 | null | null | null | null | null |
| headreset_v3_s300 | 300 / 1500 | 1.431927511264 | 1.374650061131 | 2.806577572395 | null | null |
| headreset_v3_s600 | 600 / 1800 | 1.423150670489 | 1.367088357608 | 2.790239028097 | null | null |
| coverage_v4_s200 | 200 / 2000 | 1.436555328781 | 1.349406957626 | 2.785962286407 | null | null |

`null`表示该项没有实际完成，不补零、不借父质量、不借旧速度。structure三版和headreset100只有训练/诊断，没有完整评分。headreset300/600和coverage200为完整评分负对照，没有正式计时。

正式速度仅一轮：G4pre→original→mild300→medium300→G4post，五个fresh process，同一物理A100 GPU0、原benchmark n200/warm30/seed231；FP16[1,3,512,512]均匀随机输入。原infer内行为不改，构造/编译/输入分配/warmup不计入。**512速度不能称native大图或4K速度。**

| fresh process | event p10 ms | wall p10 ms |
|---|---:|---:|
| g4_pre | 0.652288019657135 | 0.672413106076419 |
| original | 0.720896005630493 | 0.740670296363533 |
| mild | 0.719871997833252 | 0.739984004758298 |
| medium | 0.722944021224976 | 0.743143749423325 |
| g4_post | 0.779263973236084 | 0.799686741083860 |

物理UUID `GPU-823adbe3-cd4b-354f-3797-a941e3cabc9f`。五组共1000 event及1000 wall原样本，本次文档整理重算p10与已冻结证明吻合。共同normalizer在测量前固定为两G4 event p10的min：pre=0.652288019657135 ms、post=0.779263973236084 ms，取0.652288019657135 ms。

原predictor公式：g=t_p10×0.431/r，r=min(G4pre_p10,G4post_p10)；estimate=152.95×2^(S_quality−4.0841)×((0.49+2.17×0.431)/(0.49+2.17×g))^0.2。原包复现153.3691、mild300=118.2644、medium300=122.1644均是本地工具估分，**不是官方榜单分数**；153不是本轮新训练成果。不得把这些速度/估分移给600或v2/v3/v4。

正式proof：`/Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/pixg5i_hypir_distill_20261015/calibration/pixg5i_s300_formal_20261015/s300_formal_cpu.json`，SHA256 `754c56165dc234c967646905698343c8fd95a87649b0f748aebb215ac6f23eab`。JSON记录两端点估分、全部五个原timing文件SHA、统计、源/配置/权重身份及同卡连续性范围。

## 精确权重与恢复身份

以下为完整64位SHA。raw/resume真实路径、model_dir、marker/config/catalog/source及原始CM内容都在JSON各version内。此次文档未重读pth/resume，仅引用既有独立审计；headreset100本地marker副本未找到，显式标记marker_metadata_locally_opened=False，其计数来自实际CM，未伪称新读取marker。

| 版本 | raw SHA256 | resume SHA256 | CM SHA256 |
|---|---|---|---|
| original | `f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487` | `null` | `f47a1b5318be533c221d612a2baf99d90ee49267d4728e50bfb8e42b395bfbff` |
| mild_s100 | `6a2a7e69c24b8af8287ce91d7f8ec451149818fde312c6919d0eb42d714d1e2a` | `49c6ebeb0429b6b3c27dd0beaa94552679ea087d82f64705e15697aa1e268aaa` | `f1d5ca308f00f118a2d03bb7c87ee612c182410df1437f9c4ac52df082f5c739` |
| mild_s300 | `594c116d72b755625545ba8e00f4ee45ef2f83184ac62562501a53ccc7217a46` | `67b34c81fb140252f1b3b525ead033a6afa286d959181b76531d553bb779b0aa` | `2158e9756fe9fa9cac8a4e26e86c17b091b133fd2d81621c304de434e6e87922` |
| mild_s600 | `51dfbd335e4076009409ae7efc727db99f98bb6dc1cc76d1161dba9b75e3ffc5` | `04e1c91f36e4ba63324dd315186eaea69054698a475454ce96a6cc80da704dd7` | `f34003092be152c6f7723ed2a3232fcaa9a8ea4b8b279cfbfb74136cc762320d` |
| medium_s100 | `afa0b9a5702615be5314868e345a703f4014c29b40462cf38c9adc0e707008de` | `9eba194c7748e44534e06845160bd5248f57bd6b0aa0a593d4bc38023c71402c` | `d69d5c87ab7a6cabb5ca4fa6b0f91b1d4db3ae958d0afe76ac91e06516b25ce7` |
| medium_s300 | `005d13cb5b003d54e576678c52908181016cb9fa1746a9455465f5f2f619c458` | `bdde4b8e4783723f0eee3ed7e09f693e829dd5d444c5ba6ab716dc73a799580f` | `654f287813048468d994e9729de8d14cf68f126830a9d7967f49669e86de351b` |
| medium_s600 | `45c7f6bb414567f31b906fc996c037f31df7ce6aebb615c19183e02886950fab` | `45dc395979567bd0650eba92301a0d63507f217e9f7f7e0128043ca411e93b5e` | `1c5a7813c254ace9415486dd95fec8bf4412860f8ed7286e6989f1ab83b11b35` |
| structure_v2_s300 | `1424f32377bc5ad6975f942be4fb7718049a6b0b5b044a9bbe07f10e0cb971da` | `7a24fc2017412d111333af09281784ab88a69d4a0bbb6be140f9ce10897e6be6` | `273cd19af27ba763ccb912457346a4ce6f21447ce3abeef364a04ff4c5edaa95` |
| structure_v2_s600 | `97e6736da384a1320398995443f441ae92cdb9bf59a50a725e894e1ad9b5cd90` | `0feaebcdd5eff2c759c5e6a7a73f6b761cdd778381a502cecd8eb07a91ab3bd5` | `812c2b19021a24cbcd4de993fb8da328c85c3f4dfe9502a239a71674a55fb7c0` |
| structure_v2_s1200 | `2dc54794f3816c073c57a8e9f7a31e434ff80c0342ee5a6e2cc26506256a0edc` | `4d56a69128de070eb982ed1b003a4d9a4fb8c3f6d9e53f0ff63568cbd1a6b418` | `ef0e1b96ab8935d41856628721d9dd12f945048d3cf512f20454362521e2f9ec` |
| headreset_v3_s100 | `cc9aefbb2a543ea65a53c38595e4c90c8f6b76150b77e862ac4396646921f1b8` | `2039f35fa4482ffc53ce7f13e589b1d4c2124b37ab53f450d381133caf7b742f` | `e7ec8bafd8895c701a36962a7680d192ed1c82876ae1f3c624a90943f1f9eafc` |
| headreset_v3_s300 | `a40462e6481d188ed120a5fca1adbbf425f2ed3de5bd1f81f4f8628d788ff167` | `d522c6a05ff9124fb465cce867941c19300cdbc43ed505d89a3d0792aacbd7d3` | `ee0cbe9fedde18493a3b90db904f94daf0d6686d2915722a019f664d7e4df699` |
| headreset_v3_s600 | `24a68afe7c24c46fb60b768feea75ab5b32b2b681dc2a4be88765a0a05c54bfa` | `d37cbcfbd1254fd2838d36140c7cd51ef61abe0a5fedd85fc0115e1afdce9fdf` | `398a259dcb8bd6616b6d3043ed414eba73ef63e400bb21eceaf19f1160081a56` |
| coverage_v4_s200 | `503074449893247332df973a85f9be1f348d9f130f31a28455b62790fb50c929` | `e2d47be279fd3c688df232ec9ebb895f23587580ab4e40871cfb6fd79bc766ec` | `d897b171ebd1e94cfa730dd4c344045e58460813e3bd831f6929a00941660ebe` |


## 配置、源码与审计

JSON的training_phases完整保存runtime canonical protocol（重复catalog另在datasets），包括source_sha256、CLI完整值、dataset/LPIPS权重/parent引用和SHA。CLI stop_at是初始视觉门槛，不是最后实际步数；续训用明确--resume/--stop-at，不修改冻结CLI。训练独立proof验证raw=resume/FP32、Adam/scaler/RNG、日志loss/LR及case/crop或plan重放；质量proof不冒充重新做训练审计。

| 阶段 | trainer SHA256 | CLI SHA256 |
|---|---|---|
| mild | `d58e6dd66eac6b3f93a899a3008e48a568a9601a62b52cc3c99d73b9f1612604` | `00a1cb9b0d922ffb7d39df0c7671324ed21e623a4d5c285b18f3f28623004198` |
| medium | `d58e6dd66eac6b3f93a899a3008e48a568a9601a62b52cc3c99d73b9f1612604` | `3f48a21b1606124da35f99323e1e3e190cb5dbd384b742a2c31499c41229ff96` |
| structure_v2 | `3c2e83ce965ce4dbc33fa5306679c8e969f1fe1477d41af27c9ed1fc6834225e` | `1e0704c62cf6e4dae8993311b49138ca398a866162799d262ba70a5a1852d589` |
| headreset_v3 | `3e64743ff117de1934ba7da04f3c24e822ba8573d91e842230eeb86a1c6c05cb` | `10e6a2ffeb7293f1e8b989d1cff582a332dbcf13bbbfc5ff1e2f8968bec458f0` |
| coverage_v4 | `25624067758d5717e110ac7025c4e9eb8ee2ad8b2cfc292273581d6b5510fa08` | `6cd1004d143f37879dfbe512bb8e1fc2e5018cb5371bf7d0b30a0a7870c9ee9d` |

| 公共入口 | SHA256 |
|---|---|
| metadata adapter | `70c5813d20597e7260ea51b60ccdc98291e77f18e56c1131a17c4022ae4c143a` |
| original reference_runner | `6222ead92dc6fb24b88c0350e41191fc27a5adc2f6ba38bd383a4badb9bd8c1e` |
| PixG5i pix_model | `2f1cd3cd8311381e432138f489b01b982e08ebd5b09ecbc05ac87f7bfb577923` |
| renderer | `1b578c945ba4b33c7d6ce522151bb9c3a8d3a004eab0c5c7e802fad6ae1bb1f5` |
| scorer | `20125b94596c669f33dc99b592424a8c1e6a44ec1247d304853823cc5e399074` |
| csig_metrics | `1af7fecd0aa79ec9e8b9f914332cd414b9227338159813b7f45125676c8ba1f2` |
| benchmark | `3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995` |
| predictor | `0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1` |
| launcher | `422f7919521b85f71642ccca700fd579233a116a3556dd0d07bdbe77b1edc058` |
| CPU_join | `255738008c97dc14f4225f8c40a5b34ccbbc4ce3110cea8ccb71a31c11e3be2e` |

| 完整质量proof | SHA256 |
|---|---|
| coverage_v4_s200_full.json | `45562a7cc4742b48a4bf0248f95361feb588ba6ec52f3bb855d813c5a6441c71` |
| headreset_v3_s300_full.json | `07172da3c1535be01c28234018f27ae1dd78ea3b84bbfc2581a7809b4fdb8333` |
| headreset_v3_s600_full.json | `a2f8fa89e20a0c637945cafd901900d0d3f5b98eb09829e007f8e53e34f678d1` |
| original_full.json | `de857093aa2b9511bc5000d13bc4b041bd0a8177308fbb7aa6b1bee2fa1d5d53` |
| stage100_full.json | `2db3fc2572921f9fe2e1f450cbddf6374178ba4e2a56b28871dc83373358bb8d` |
| stage300_full.json | `08266b9f60244208a60e63465d138d537515665a7dfda7cc124087f842253886` |
| stage600_full.json | `8714a1143e7ba64c4a65a1d0baa1eb7dc92342bd790b5631c9c258f72f8f05f8` |

| 独立训练proof | SHA256 |
|---|---|
| coverage_v4_training_steps1_cpu.json | `d74c432048df500af138060d6a15bbabfb73103ae1258abe193052caffb88603` |
| coverage_v4_training_steps200_cpu.json | `7d4be677283aced016ddd69f70994bc6412274231282d5428344ddb86978a4c7` |
| headreset_v3_training_steps1_cpu.json | `572f27112e97982da122c2adfe996ccf207eeced2038f53ac1b7c572a40967d5` |
| headreset_v3_training_steps300_cpu.json | `8ee6ffa0d61562f12ba03da5cc8b497b242d5258114788785c0128f806571332` |
| headreset_v3_training_steps600_cpu.json | `067ccfef156437a0cb4fe3e002822d267fee9aab9a7c00f375ae86c37b46d1cc` |
| structure_v2_training_steps1200_cpu.json | `32cad4985898599f4ebb1d1579401e5a466185a055c44d64924dd83deab14975` |
| structure_v2_training_steps1_cpu.json | `df99cbb850c17ab14b1e59a64dd5bb4ba0c57d1d737be0231ae6e78c5e171b2e` |
| structure_v2_training_steps300_cpu.json | `42190023f32276cc19b4e0acf626cb38dd76c348a4acc7165c9b958f3a58a14e` |
| structure_v2_training_steps600_cpu.json | `9ba2cae554c987025acfd4f2b51095a556a09f8a3659b60481e153e667a12222` |
| training_steps1_100_cpu.json | `9e0bc526300420cf270af046a171729adffcb17eb6b186c5f036ae470a921cbc` |
| training_steps300_cpu.json | `ffa15ae1d69e6c30d0b2edb65b04f41760fcafa82fb88b5d69f716f27eba45cc` |
| training_steps600_cpu.json | `08f4852e984dc97d26abc6a3ea734cabe9547f512aa4d8ca0cc08adfa4331b05` |


计分环境由proof记录：Torch2.8.0+cu128、torchvision0.23.0+cu128、pyiqa0.1.10、numpy2.3.3、Pillow11.3.0、timm1.0.28；其余逐包版本在JSON quality_protocol.environment。metric weights引用恢复证明SHA256 `dbab8b00ca223d101f61a5154d1a161256afe9d367ba18da74de6c1360cd5947`，不改原指标公式/权重。

复现说明（只作入口记录，不在此次文档任务执行）：先从已验证备份恢复固定输入/教师/源码/权重/环境，按JSON核对完整SHA；使用 `R/env/bin/python C/train_*.py --config C/config_*.json --validate-config-only` 检查配置；从新目录开始或用明确immutable resume，按同phase --stop-at恢复。代码内parent/source/protocol路径和SHA有严格门槛，不能只改根目录便假定同一实验。注册器为普通cfe、structure c396、headreset3a2、coverage6bd；源SHA在各CM registration_source。部署不可覆盖，从真实marker登记权重，不预注册不存在文件。原103renderer及418scorer仍为上表固定源。

## 视觉门槛、最终失败结论与保留

既有7/16/41/83原尺寸1:1图库和root/strategy实际观测：密集虫纹和部分孔斑减少，但字形鼓包、栏杆融接/粗化、叶片/背景结构差距与平滑仍在；structure阶段有局部拖影；headreset接近柔糊输入。这里是记录既有视觉结论，没有在文档阶段重新凭无GT判定纹理真假。

最新coverage200：NR=1.436555328781106、FR=1.349406957626343、S_quality=2.785962286407448；相对medium600 ΔS=-0.911130377018867，相对headreset600 ΔS=-0.004276741689446。完整性审计PASS只代表真实图数/来源/公式闭合，不表示选用或视觉合格。未续600，不跑正式速度，不更换main。

最终状态来源：`/Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/pixg5i_hypir_distill_20261015/continuation_20261015_result.json`。图库 `/Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/pixg5i_hypir_distill_20261015/visuals/coverage_v4_s200_vs_medium300_teacher_diagnostic/index.html`；diag与full四例SHA/ROI像素等价证明 `/Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/pixg5i_hypir_distill_20261015/visuals/coverage_v4_s200_vs_medium300_teacher_full/full_diag_equivalence.json`。每个版本的本地backup_*_verified/artifact保留范围由其manifest/receipt界定，不能推断所有完整PNG均已备份。本次GitHub小文档不打包模型、resume、整套数据或大图。

机器清单SHA256：`c683f7af88bd1891b177b029a354b7bbe9059afe65bab953c2a29e877896f15d`。source_summaries记录两个结果文件的实际SHA；quality_proofs/training_audits与各CM均留完整来源。原始proof、历史结果和HYPIR指针未改。
