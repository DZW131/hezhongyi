# 环境记录

执行日期为2026-11-11。服务器为IPMI-System-Product-Name，Python 3.10.20 (main, Jun 11 2026, 15:17:37) [GCC 14.3.0]，PyTorch 2.11.0+cu128，CUDA运行时12.8。

GPU检查时已有3个计算进程，共占用16776 MiB。实验采用CPU、4线程、FP32。GPU显存峰值记为N/A，GPU推理未执行。项目虚拟环境通过sshr5090的Python创建并继承其只读依赖，仅在项目.venv安装imgaug 0.4.0。

磁盘可用空间初始为131 GB。详细驱动、进程和磁盘记录在logs/environment.json。

官方代码提交为71a6298538d3af2c2fc2983f078e6290c66e59d3。服务器直连GitHub发生证书校验失败，使用本地从官方HTTPS克隆的git bundle传输，同一提交已核对。上游保存在upstream，兼容副本位于compat_cpu。

官方权重为OrderedDict，141个键，33973845字节。SHA-256为c7a042638cd44fa44533e6ab621cc19b4bf60cfcf620fda9e8e12cbd5ae7d3b5，本地和服务器一致。模型strict=True加载成功，参数量8479114。加载使用weights_only=True。

代码许可证为上游GPL-2.0。权重和Docker镜像的单独条款仍待核实。权重来源为官方README的Google Drive链接，记录见configs/run_metadata.json。
