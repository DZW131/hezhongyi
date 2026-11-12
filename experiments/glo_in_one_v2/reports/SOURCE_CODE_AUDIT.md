# 源码与兼容审计

真实调用链为Glo_v2_segmentation.py → MOTSDataset_2D_Patch_normal_save_csv_Glo_v2.py生成data_list.csv → testing_2D_patch_v2.py → MOTSValDataSet_normal → unet2D_ns_for_Glo_v2.UNet2D → 15次task条件前向 → 每任务比较两类logit → 按task 0到14拼接保存。

代码位置均指upstream原件。

- Glo_v2_segmentation.py末尾两个subprocess.run未检查退出码。兼容副本增加check=True，并用sys.executable保证子进程环境一致。
- CSV生成文件第128–130行将task/scale均设为0，第167行将CSV写入输入目录。实验只传入独立staging副本。
- MOTSDataset_2D_Patch_supervise_normal_csv_Glo_v2.py第503–527行使用plt.imread，取前三通道并转float32 CHW。PNG输入为RGB [0,1]，不减均值、不除标准差，无缩放、padding、裁剪或验证增强。命令行input_size=512,512未在该类实施resize。
- testing_2D_patch_v2.py第121–125行默认input_size=512,512、batch_size=4、FP16=False。外层DataLoader batch=1，内部ImagePool按4图分组，尾组走剩余分支。两个官方输入均为512方图。
- 第597–611行依次调用task 0–14。第706–716行拼接为H×W×15，前景比较为logit[1]>logit[0]。尾组同样保存15通道。各类允许重叠，未做argmax。
- 模型第158–172行编码15维task与4维scale。scale=0的物理倍率含义缺少来源确认。
- 模型中硬编码CUDA分配，CPU兼容副本只将设备分配改为CPU。未改变网络层、参数、task、scale和阈值。
- FP32单进程推理用不到Apex训练/分布式功能，兼容副本移除相应导入。删除两个数据文件中未使用的nibabel/SimpleITK导入，将TkAgg改Agg，DataLoader workers设0。完整diff见cpu_compat.patch。

requirements.txt是Linux conda导出规格。当前采用项目独立venv继承既有环境，避免直接pip安装该文件。未执行上游训练脚本。

两张demo重复推理逐字节一致。CPU与CUDA输出等价性尚未实测，必须与部署性能区别记录。
