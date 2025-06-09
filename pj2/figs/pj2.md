# pj2
21307140107 陈颂若
# 1. Train a Network on CIFAR-10
## 1.简单CNN
* 模型结构为（参数量：62006）:

| 层  | 输出形状  |
|:----------|:----------|
| Conv2d+relu  | (6, 28, 28)    |
| MaxPool   |(6, 14, 14)|
| Conv2d+relu   | (16,10,10)  |
| MaxPool       |(16,5,5) |
| Flatten   |(400,)  |
| Linear+relu   | 120   |
| Linear+relu  | 84|
| Linear   | 10   |
 
* 训练配置及结果：

|优化器|调度器|轮数|测试acc|lr|
|:-----|:-----|:------|:-----|:--|
|SGD|None|200|64%|0.001|

* 训练图线:
![loss曲线](simple_loss.png)
![acc曲线](simple_acc.png)

* 总结:
简单CNN训练参数较少，训练起来非常快(平均约4.89s/epoch)，但是模型层数较浅，训练多轮（100）性能也不够高。
* 权重文件：
[link](https://drive.google.com/file/d/1jVbfX_tetB7FLBB7FBq5TUvI9UbevfKS/view?usp=drive_link)
## 2.加入残差连接+batch norm层
* 模型结构为（参数量：692122）:

| 编号 | 层类型                            |
|--------|-----------------------------------|
| 1      | Conv2d(3→16, 3×3, padding=1) + BatchNorm + ReLU |
| 2      | ResidualBlock1: Conv(16→32, 3×3, stride=2) + BN + ReLU → Conv(32→32, 3×3) + BN + downsample(1×1 Conv) + skip |
|        | ResidualBlock2: Conv(32→32, 3×3) + BN + ReLU → Conv(32→32, 3×3) + BN + skip |
| 3      | ResidualBlock3: Conv(32→64, 3×3, stride=2) + BN + ReLU → Conv(64→64, 3×3) + BN + downsample(1×1 Conv) + skip |
|        | ResidualBlock4: Conv(64→64, 3×3) + BN + ReLU → Conv(64→64, 3×3) + BN + skip |
| 4      | ResidualBlock5: Conv(64→128, 3×3, stride=2) + BN + ReLU → Conv(128→128, 3×3) + BN + downsample(1×1 Conv) + skip |
|        | ResidualBlock6: Conv(128→128, 3×3) + BN + ReLU → Conv(128→128, 3×3) + BN + skip |
| 5      | AdaptiveAvgPool2d((1, 1))         |
| 6      | Flatten                           |
| 7      | Linear(128 → 10)                  |

* 训练配置和结果：

|opti|sche|epoch|acc|lr|
|:---|:---|:---|:---|:--|
|adam|CosineAnnealingLR|100|77.06%|1e-4|

* 训练图线:

![loss曲线](deep_loss.png)
![acc曲线](deep_acc.png)

* 总结:
具有残差连接和BN层的卷积网络有更优越的性能，收敛速度很快（58轮则达到最优测试性能），残差连接和BN层是非常有效的
* 权重文件：
[link](https://drive.google.com/file/d/1s6cHtYxorG6nXae1XHuixqvBHUnGk23z/view?usp=drive_link)
## 3.更深的网络
* 模型结构：在之前基础上加深了模型，它有**9**个残差连接块，共**3**层，每层3个。（参数量：1080090）

| 编号 | 层类型                                     |
|--------|--------------------------------------------|
| 1      | Conv2d(3→16, 3×3, padding=1) + BatchNorm + ReLU |
| 2      | ResidualBlock1_1: Conv(16→32, 3×3, stride=2) + BN + ReLU → Conv(32→32, 3×3) + BN + downsample(1×1 Conv) + skip |
|        | ResidualBlock1_2: Conv(32→32, 3×3) + BN + ReLU → Conv(32→32, 3×3) + BN + skip |
|        | ResidualBlock1_3: Conv(32→32, 3×3) + BN + ReLU → Conv(32→32, 3×3) + BN + skip |
| 3      | ResidualBlock2_1: Conv(32→64, 3×3, stride=2) + BN + ReLU → Conv(64→64, 3×3) + BN + downsample(1×1 Conv) + skip |
|        | ResidualBlock2_2: Conv(64→64, 3×3) + BN + ReLU → Conv(64→64, 3×3) + BN + skip |
|        | ResidualBlock2_3: Conv(64→64, 3×3) + BN + ReLU → Conv(64→64, 3×3) + BN + skip |
| 4      | ResidualBlock3_1: Conv(64→128, 3×3, stride=2) + BN + ReLU → Conv(128→128, 3×3) + BN + downsample(1×1 Conv) + skip |
|        | ResidualBlock3_2: Conv(128→128, 3×3) + BN + ReLU → Conv(128→128, 3×3) + BN + skip |
|        | ResidualBlock3_3: Conv(128→128, 3×3) + BN + ReLU → Conv(128→128, 3×3) + BN + skip |
| 5      | AdaptiveAvgPool2d((1, 1))                  |
| 6      | Flatten                                    |
| 7      | Linear(128 → 10)                           |

* 训练配置和结果：

|opti|sche|epoch|acc|lr|
|:---|:---|:---|:---|:--|
|adam|CosineAnnealingLR|180|**90.29%**|1e-4|

* 训练图像:

![loss曲线](deeper_loss.png)
![acc曲线](deeper_acc.png)
* 总结:
上一个模型已经证明了残差连接和BN的有效性，具有更多层同结构的网络展现了优越的性能。

* 权重文件：
[link](https://drive.google.com/file/d/16Q9HZThnWu_4UtUbdnRMGIk2s5ZOwTvg/view?usp=drive_link)

* 可视化特征图

![feature](feature.png)

# 2. Batch Normalization
* VGGnet loss landscape
![loss_landscape](loss_landscape.png)

* 从图中看到，使用了batch norm的vgg的训练过程中更为平稳，对初始学习率的敏感性较低，并且熟练速度更快，具有更小的loss。
 
* lr=5e-4时 vgg_A的训练曲线：

![fig](loss_curve.png)

![fig](val_accuracy_curve_epoch.png)

权重：[link](https://drive.google.com/file/d/1XatcAA1_yhOYrMcCMFsL2auIyPWxnGqi/view?usp=drive_link)

* lr=5e-4时 vgg_A with batchnorm 的训练曲线：

![fig](loss_curve_B.png)

![fig](val_accuracy_curve_epoch_B.png)

权重：[text](https://drive.google.com/file/d/1hetqDOjKNC3QfJKyXR2JFZdYFHkHgarJ/view?usp=drive_link)

* 对比上图可见，有了batchnorm的vggnet 收敛更快更好，测试性能好。
* batchnorm平滑loss landscape,这种平滑性带来了更具可预测性的梯度，从而允许使用更宽范围的学习率，并加快了网络收敛速度。

code:

