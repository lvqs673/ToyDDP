

**实现一个简单的分布式数据并行**
- 使用环形拓扑，三台主机
- 梯度的同步通过reduce-scatter和allgather实现


**实验任务使用LSTM进行时序预测**
- 时序数据为100个糖尿病患者的血糖变化
- ./data/blood_glucose目录下有100位糖尿病患者的血糖数据，每个文件是一个患者的血糖数据
   - 每个文件中：第一列是血糖监测点编号，第二列是血糖浓度（单位mmol/L）
   - 每个监测点代表5分钟内的血糖浓度均值
- 实验预测目标是对未来1小时内的血糖浓度进行预测，因此需要预测12个监测点的值
- 每次预测参考前边10个小时的血糖变化情况，也就是前边120个监测点
***所以这个任务做的东西就是输入120个数据，输出12个数据***


**数据集构造**
1. 首先需要将序列数据转换为监督型数据。可以取大小为132的窗口在这个序列上滑动，这样可以得到若干条132维向量。
2. 对100个患者的序列数据均做如上处理，可以得到很多132维向量，将这些向量合并得到总的监督型数据。
3. 按照7:3划分训练集和测试集。

**模型**
- 模型包括两层，一层LSTM层和一层Dense层
    - LSTM层：使用两层的BiLSTM，隐藏层维度设置为100
    - Dense层：将LSTM层的输出映射到12维
- 模型输入：120维的向量
- 模型输出：12维的向量

**实验中记录的指标**
- 训练集上每个epoch的MSE、MAE（绘制Loss-Epoch曲线可视化，将单&多机的MSE绘制在一张图、MAE绘制在另一张图）（单机+多机）
- 测试集上每个epoch的MSE、MAE（绘制Loss-Epoch曲线可视化）（单机+多机）
- 对第100个患者的预测值（将该患者的实际血糖浓度和单机&多机的模型的预测血糖浓度绘制在同一张图中进行对比）（单机+多机）
- 单机下总的训练耗时；多机下总训练耗时、计算耗时、通信耗时：计算多机的加速比、有效计算时间（计算耗时/总训练时间）



