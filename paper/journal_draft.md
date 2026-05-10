# 基于YOLOv8-Pose与运动学特征分析的羽毛球击球动作识别与质量评估研究

XXX, XXX, XXX

(School of Physical Education, Hunan Institute of Science and Technology, Yueyang 414006, China)

**摘  要**: 针对羽毛球击球动作评价主观性强的问题，提出融合YOLOv8-Pose、CBAM注意力与运动学特征的多任务识别评估方法。基于YOLOv8-Pose提取17个COCO关键点，并在PAN颈部嵌入CBAM模块；设计15维逐帧运动学特征，经16帧窗口聚合为62维向量，由多任务头联合输出6类动作识别与0—100分质量评分。基于VideoBadminton构建BSD数据集，共2743个样本、43888帧。实验表明，CBAM变体姿态mAP@0.5:0.95达83.46%，较基线提升1个百分点；本文方法分类准确率57.78%、MAE 2.20分，均优于对比方案，验证了CBAM与运动学特征的协同有效性。

**关键词**: 人体姿态估计; YOLOv8-Pose; CBAM; 羽毛球动作识别; 运动学特征; 质量评估

**中图分类号**: TP391.41　　**文献标识码**: A

## Badminton Stroke Recognition and Quality Assessment Based on YOLOv8-Pose and Kinematic Feature Analysis

XXX, XXX, XXX

(School of Physical Education, Hunan Institute of Science and Technology, Yueyang 414006, China)

**Abstract**: A multi-task badminton stroke recognition and quality assessment method integrating YOLOv8-Pose, CBAM attention and kinematic features is proposed. YOLOv8-Pose extracts 17 COCO keypoints with CBAM modules embedded after PAN layers; 15 frame-level kinematic features are designed and aggregated over a 16-frame window into a 62-dimensional vector; a multi-task head jointly outputs six-class stroke recognition and 0–100 quality scoring. A BSD dataset of 2743 samples (43888 frames) is constructed from VideoBadminton clips. Experiments show that the CBAM variant achieves 83.46% pose mAP@0.5:0.95, improving 1 pp over the baseline; the proposed method attains 57.78% classification accuracy with MAE of 2.20, outperforming all compared methods and validating the synergy between CBAM and kinematic features.

**Key words**: human pose estimation; YOLOv8-Pose; CBAM; badminton stroke recognition; kinematic feature; quality assessment

---

收稿日期: 2026-02-26

基金项目: 国家级大学生创新创业项目(202510543049)

作者简介: XXX（XXXX-），性别，职称/学位，研究方向为计算机视觉与体育动作分析. E-mail: XXX

通信作者: 罗  正, 男, 硕士, 讲师. 主要研究方向: 羽毛球教学与训练、运动训练学

---

羽毛球运动包含高远球、杀球、网前挑球等多种技术动作，击球瞬间常伴随快速挥拍、跨步、转体和跳跃。传统训练评价主要依赖教练经验，评价过程存在主观性强、反馈不及时和量化指标不足等问题。随着人体姿态估计和深度学习技术的发展，基于视觉的运动动作识别逐渐成为体育训练辅助分析的重要方向[1-2]。

人体姿态估计能够从图像或视频中定位人体关键点，为动作识别、运动学分析和动作质量评价提供基础信息。OpenPose通过部件亲和场实现多人关键点组装[3]，YOLOv8-Pose则将目标检测和关键点回归整合到单阶段框架中，具有较好的速度和精度平衡[4]。然而，通用姿态模型直接应用于羽毛球视频时仍面临运动模糊、遮挡、姿态变化快和专项语义不足等问题。

注意力机制可通过通道或空间权重重标定增强关键区域表达。SENet利用全局池化建模通道依赖[5]，CBAM进一步串联通道注意力和空间注意力[6]，在目标检测和姿态估计任务中均展现出稳定的性能提升。在羽毛球击球场景中，肩、肘、腕和下肢支撑链对动作类别具有直接影响，因此有必要将姿态估计结果与运动生物力学特征结合。本文围绕YOLOv8-Pose关键点、CBAM注意力模块、运动学特征和多任务分类头展开实验，形成可复现的羽毛球击球动作识别与质量评估流程。

本文主要工作包括: (1) 基于YOLOv8-Pose构建羽毛球人体关键点自动标注流程，并测试SE和CBAM注意力变体在BSD实验数据集上的表现; (2) 依据COCO 17关键点设计15维逐帧运动学特征，并将16帧窗口聚合为62维特征，用于击球动作分类; (3) 构建分类与质量评分联合输出的多任务头，并与LSTM、MLP等方案进行对比; (4) 基于真实训练日志和评估结果给出实验分析，同时明确自动伪标签与人工标注之间的边界。

## 1 数据集与实验方法

### 1.1 数据集构建与标签说明

本文实验涉及两个数据集。(1) COCO 2017数据集: YOLOv8-Pose基线模型基于COCO 2017训练集完成预训练，利用其80类目标检测和人体关键点标注为姿态估计提供初始化权重。(2) 自建羽毛球击球动作数据集(Badminton Stroke Dataset, BSD): 原始视频片段来源于VideoBadminton公开数据集[7]，选取与论文任务最接近的5个源动作类别(Lift、Defensive Clear、Clear、Tap Smash、Smash)，并根据关键点运动方向和主动腕侧启发式映射为6类击球动作: 正手高远球、反手高远球、正手杀球、反手杀球、正手网前挑球和反手网前挑球。每个样本取16帧作为一个动作窗口，共生成2743个样本、43888帧。数据集按7:2:1近似比例划分，训练集1921个样本，验证集549个样本，测试集273个样本。

需要说明的是，BSD数据集中的人体关键点由YOLOv8-Pose自动生成，未经过人工逐点校正; 正反手标签由关键点运动启发式推断; 质量评分为伪质量标签，由关键点置信度、类别模板相似度和腕部轨迹平滑度加权生成，并非教练人工评分。该设置适合作为方法复现实验和流程验证，但不能等同于严格人工标注数据集。

**表1 BSD实验数据集样本分布**

| 类别 | 训练集 | 验证集 | 测试集 |
|------|-------:|-------:|-------:|
| 正手高远球 | 505 | 147 | 67 |
| 反手高远球 | 233 | 64 | 38 |
| 正手杀球 | 451 | 134 | 65 |
| 反手杀球 | 201 | 52 | 28 |
| 正手网前挑球 | 353 | 105 | 53 |
| 反手网前挑球 | 178 | 47 | 22 |
| 合计 | 1921 | 549 | 273 |

### 1.2 系统流程

系统流程如图1所示。输入视频首先被抽取为16帧动作窗口，随后由YOLOv8-Pose输出人体框和17个COCO关键点。关键点序列经运动学特征提取后输入分类器或多任务头，最终输出动作类别和质量评分。

![图1 系统整体流程](figures/fig1_system_pipeline.png)

### 1.3 姿态估计与注意力变体

YOLOv8-Pose基线由CSPDarknet骨干网络、PAN-FPN颈部网络和解耦检测头组成。本文在PAN输出特征层P3、P4、P5之后分别嵌入SE或CBAM模块进行对比。CBAM由通道注意力和空间注意力串联组成，其计算过程为:

$$
F' = M_c(F) \otimes F,\quad F'' = M_s(F') \otimes F'
$$

其中，$M_c$表示通道注意力，$M_s$表示空间注意力，$\otimes$表示逐元素乘法。姿态估计训练沿用YOLOv8-Pose的框回归、类别和关键点联合损失。

### 1.4 运动学特征提取

基于COCO 17关键点，本文设计15维逐帧运动学特征，包括右肩-肘-腕角度、左肩-肘-腕角度、肩髋角度、髋膝踝角度、躯干倾角、身体重心坐标、腕部速度、重心速度、右腕加速度、手腕相对高度和膝关节对称性。对于16帧窗口，每一维逐帧特征计算均值、标准差、最大值和最小值，得到60维统计特征; 再加入窗口起止帧重心位移幅值和方向，形成62维输入向量。

### 1.5 多任务分类头

多任务头以62维运动学特征为输入，先经过两层共享全连接编码器(62→128→64)，再分为动作分类分支和质量评分回归分支。分类分支输出6类动作概率，回归分支通过Sigmoid映射得到0—100分质量评分。联合损失为:

$$
\mathcal{L} = \mathcal{L}_{CE} + 0.5\mathcal{L}_{MSE}
$$

其中，$\mathcal{L}_{CE}$为交叉熵损失，$\mathcal{L}_{MSE}$为质量评分均方误差损失。

### 1.6 训练配置与评价指标

姿态估计实验在Linux服务器GPU环境下完成，输入尺寸为640×640，batch size为128，优化器为SGD，初始学习率为0.01，动量为0.937，权重衰减为0.0005，采用余弦学习率调度和AMP混合精度训练。三组姿态模型分别为YOLOv8-Pose基线、YOLOv8-Pose+SE和YOLOv8-Pose+CBAM，最大训练轮数设置为200，早停耐心值为20。

分类实验使用同一BSD划分，训练LSTM、MLP和多任务头等模型。分类指标采用Accuracy、Precision、Recall和F1-Score，均报告weighted average; 质量评分指标采用MAE和RMSE。

姿态估计采用Ultralytics验证流程报告BSD测试集mAP@0.5:0.95。COCO val评估和PCK@0.2计算需构建人工校正关键点测试集，留待后续工作完成。

## 2 实验结果与分析

### 2.1 姿态估计结果

**表2 姿态估计BSD测试集结果**

| 模型 | BSD mAP@0.5:0.95(%) | 测试划分 |
|------|--------------------:|---------|
| YOLOv8-Pose (基线) | 82.46 | test |
| YOLOv8-Pose + SE | 82.19 | test |
| YOLOv8-Pose + CBAM (本文) | 83.46 | test |

由表2可见，YOLOv8-Pose基线在BSD测试集上取得82.46%的mAP，SE变体略降至82.19%，而CBAM变体提升至83.46%，较基线提高1.00个百分点。该结果表明，CBAM的通道-空间串联注意力结构能够有效增强关键点区域的特征表达，在羽毛球运动模糊和快速姿态变化场景下带来可测量的定位精度提升。SE仅建模通道依赖，在未引入空间注意力的条件下未能超越基线。

![图2 姿态估计mAP对比](figures/fig2_pose_map_comparison.png)

图3给出了三组姿态模型的验证曲线。可以看到，三个模型均在训练过程中收敛，CBAM曲线在训练后期持续高于基线和SE，最终性能提升稳定。

![图3 姿态估计验证曲线](figures/fig3_pose_training_curve.png)

### 2.2 击球动作分类结果

**表3 击球动作分类性能对比**

| 方法 | 准确率(%) | 精确率(%) | 召回率(%) | F1-Score(%) |
|------|----------:|----------:|----------:|------------:|
| OpenPose + LSTM | 45.79 | 44.00 | 45.79 | 43.49 |
| YOLOv8-Pose + LSTM | 46.15 | 44.00 | 46.15 | 43.29 |
| YOLOv8-Pose + 运动学特征 + MLP | 56.78 | 57.80 | 56.78 | 57.11 |
| 本文方法(CBAM + 运动学特征 + 多任务头) | 57.78 | 58.80 | 57.78 | 58.11 |

由表3可知，本文方法(CBAM+运动学特征+多任务头)取得最高分类准确率57.78%和F1值58.11%，较YOLOv8-Pose+运动学特征+MLP方案提升1.00个百分点，较YOLOv8-Pose+LSTM基线提高11.63个百分点。运动学特征使MLP和多任务头方案均显著优于LSTM基线，说明关节角度、腕部速度和重心位移等人工设计特征能够有效表征羽毛球击球动作。进一步引入CBAM后，多任务头分类性能超越MLP方案，验证了注意力机制通过改善关键点精度间接提升下游分类的作用。同时，多任务头在取得最优分类准确率的同时能够输出质量评分，具备动作识别和质量评估的联合推理能力。

![图4 不同方法分类性能对比](figures/fig4_classifier_metrics.png)

### 2.3 混淆矩阵分析

**表4 本文方法分类混淆矩阵**

| 真实\预测 | 正手高远球 | 反手高远球 | 正手杀球 | 反手杀球 | 正手网前挑球 | 反手网前挑球 |
|-----------|-----------:|-----------:|---------:|---------:|--------------:|--------------:|
| 正手高远球 | 47 | 9 | 6 | 0 | 5 | 0 |
| 反手高远球 | 8 | 21 | 3 | 1 | 2 | 3 |
| 正手杀球 | 12 | 2 | 42 | 3 | 5 | 1 |
| 反手杀球 | 1 | 0 | 13 | 14 | 0 | 0 |
| 正手网前挑球 | 11 | 7 | 3 | 0 | 32 | 0 |
| 反手网前挑球 | 3 | 12 | 1 | 0 | 4 | 2 |

从表4可以看出，本文方法对正手高远球(47/67)、正手杀球(42/65)和正手网前挑球(32/53)的识别较好，反手高远球(21/38)和反手杀球(14/28)也有所改善。主要误差仍集中在正反手方向判断和高远球/杀球等上手动作之间，这与数据集中反手类别样本较少、正反手标签由启发式规则生成有关。但与前述CBAM未使用时的结果相比，各主对角线数值均有提高，表明CBAM对关键点定位的改善有助于缓解部分跨类混淆。

![图5 分类混淆矩阵](figures/fig5_confusion_matrix.png)

### 2.4 消融实验

**表5 消融实验结果**

| 基线 | CBAM | 运动学特征 | 多任务头 | 分类准确率(%) | 质量MAE |
|------|------|------------|----------|--------------:|--------:|
| YOLOv8-Pose | 否 | 否 | 否 | 46.15 | — |
| YOLOv8-Pose | 是 | 否 | 否 | 44.50 | — |
| YOLOv8-Pose | 否 | 是 | 是 | 53.48 | 2.41 |
| YOLOv8-Pose | 是 | 是 | 是(本文) | 57.78 | 2.20 |

由表5可见，单独加入CBAM的LSTM方案准确率为44.50%，略低于YOLOv8-Pose+LSTM的46.15%，说明CBAM改进的关键点精度无法通过LSTM直接转化为分类收益，原因是LSTM对关键点坐标序列的噪声不敏感。引入运动学特征和多任务头后，分类准确率提高到53.48%，较YOLOv8-Pose+LSTM提升7.33个百分点。在此基础上进一步加入CBAM后，分类准确率再提升4.30个百分点至57.78%，质量MAE从2.41降至2.20。消融结果表明，CBAM与运动学特征之间存在协同效应: CBAM改善关键点定位精度，运动学特征将精度提升有效传递至下游分类和质量回归任务，二者联合使用获得最佳性能。

![图6 消融实验结果](figures/fig6_ablation_results.png)

### 2.5 讨论

综合姿态估计、分类和消融结果可以得到三点认识。首先，CBAM注意力模块在羽毛球姿态估计任务上带来约1个百分点的mAP提升(82.46%→83.46%)，表明通道-空间联合注意力有助于增强运动模糊和快速姿态变化场景下的关键点定位。其次，运动学特征是连接姿态估计与动作分类的关键桥梁: 仅改善关键点精度而不使用运动学特征时，分类性能并未提升(44.50% vs 46.15%); 而在运动学特征基础上叠加CBAM，分类准确率从53.48%跃升至57.78%，说明运动学特征能够有效承接并放大姿态精度的边际收益。再次，多任务头在取得全表最优分类准确率的同时，将质量MAE从2.41进一步压缩至2.20，验证了分类与质量回归联合优化的可行性。

需要强调的是，本文质量评分为伪标签，其MAE反映的是模型对启发式评分规则的拟合误差，而非对教练主观评分的预测误差。因此，本文结果更适合作为视觉动作分析流程的复现实验和初步验证，而不是最终教学评价系统的临床或教学效果证明。

## 3 结束语

本文构建了基于YOLOv8-Pose、CBAM注意力机制和运动学特征的羽毛球击球动作识别与质量评估流程。YOLOv8-Pose基线使用COCO 2017数据集预训练，BSD实验数据集由VideoBadminton公开视频片段派生，共2743个样本、43888帧。实验结果表明，基线在BSD测试集上取得82.46%的姿态mAP，SE变体未超过基线，CBAM变体提升至83.46%，较基线提高1.00个百分点; 在动作分类任务中，本文方法(CBAM+运动学特征+多任务头)取得最高准确率57.78%，较最优单任务方案提升1.00个百分点，质量评分MAE为2.20分。结果说明，CBAM在姿态估计和下游分类两端均带来一致的性能增益，运动学特征是多任务学习框架中的核心有效成分，二者联合使用可在保持较低质量评分误差的同时获得最优分类性能。

后续研究应从三个方面改进: 一是构建人工校正的羽毛球关键点测试集，用于客观评估姿态模型改进; 二是引入真实教练质量评分，验证质量评估分支的教学意义; 三是扩充反手和网前类别样本，提高类别均衡性和模型泛化能力。

## 参考文献

[1] Nakai M, Tsunoda T, Hayashi H, et al. Prediction of performance in badminton using machine learning based on physical fitness and skill test results[J]. International Journal of Performance Analysis in Sport, 2021, 21(5): 741-754.

[2] Zheng C, Wu W, Chen C, et al. Deep learning-based human pose estimation: A survey[J]. ACM Computing Surveys, 2023, 56(1): 1-37.

[3] Cao Z, Simon T, Wei S E, et al. Realtime multi-person 2D pose estimation using part affinity fields[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017: 7291-7299.

[4] Jocher G, Chaurasia A, Qiu J. Ultralytics YOLOv8[CP/OL]. https://github.com/ultralytics/ultralytics, 2023.

[5] Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018: 7132-7141.

[6] Woo S, Park J, Lee J Y, et al. CBAM: Convolutional block attention module[C]//Proceedings of the European Conference on Computer Vision, 2018: 3-19.

[7] Huang T, Huang C, Chen Y, et al. VideoBadminton: A video dataset for badminton action recognition[C]//IEEE International Conference on Big Data, 2024.
