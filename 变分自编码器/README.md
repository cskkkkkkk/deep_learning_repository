变分自编码器生成MNIST  手写数字（结合代码描述实现步骤以及提交下面要求提交的结果）
推荐使用高斯分布随机初始化模型参数，可以避免一部分模式坍塌问题。

1、模型架构：

① 编码器（全连接层）： 输入图片维度：784 (28 × 28) 输出层维度（ReLU）：400 
② 生成均值（全连接层）： 输入层维度：400
输出层维度：20
③ 生成标准差（全连接层）： 输入层维度：400
输出层维度：20
④ 使用均值和标准差生成隐变量z

⑤ 解码器（全连接层）： 输入维度：20
隐藏层维度（ReLU）：400输出层维度（Sigmoid）：784
训练完网络，需要提交重构损失和KL散度的随迭代次数的变化图，以及10 张生成的手写数字图片。
