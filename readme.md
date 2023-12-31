

**实现一个简单的分布式数据并行**
- 使用环形拓扑，三台主机
- 梯度的同步通过reduce-scatter和allgather实现


**实验任务使用LSTM进行时序预测**
- 时序数据为100个糖尿病患者的血糖变化
- *./data/blood_glucose*目录下有100位糖尿病患者的血糖数据，每个文件是一个患者的血糖数据
   - 每个文件中：第一列是血糖监测点编号，第二列是血糖浓度（单位mmol/L）
   - 每个监测点代表5分钟内的血糖浓度均值
- 实验预测目标是对未来一小时内的血糖浓度进行预测，因此需要预测12个监测点的值
- 每次预测参考前边10个小时的血糖变化情况，也就是前边120个监测点
- ***所以这个任务做的事情就是输入120个数据，输出12个数据***


**数据集构造**
1. 首先需要将序列数据转换为监督型数据。可以取大小为132的窗口在这个序列上滑动，这样可以得到若干条132维向量。
2. 对100个患者的序列数据均做如上处理，可以得到很多132维向量，将这些向量合并得到总的监督型数据。
3. 按照7:3划分训练集和测试集。

**模型**
- 模型包括两层，一层LSTM层和一层Dense层
    - LSTM层：使用两层的BiLSTM，隐藏层维度设置为200
    - Dense层：将LSTM层的输出映射到12维
- 模型输入：120维的向量
- 模型输出：12维的向量

**实验中记录的指标**
- 训练集上每个epoch的MSE、MAE（绘制Loss-Epoch曲线可视化，将单&多机的MSE绘制在一张图、MAE绘制在另一张图）（单机+多机）
- 测试集上每个epoch的MSE、MAE（绘制Loss-Epoch曲线可视化）（单机+多机）
- 对第100个患者的预测值（将该患者的实际血糖浓度和单机&多机的模型的预测血糖浓度绘制在同一张图中进行对比）（单机+多机）
- 单机下总的训练耗时；多机下总训练耗时、计算耗时、梯度同步耗时：计算多机的加速比、有效计算时间（计算耗时/总训练时间）

**代码实现**
- *config.py*: 部分配置信息
- *dataset.py*: 血糖预测数据集相关
- *model.py*: 模型相关
- *utils.py*: 用到的一些通用函数
- *train_sn.py*: 单结点的训练脚本（sn为Single Node）
- *train_sn.sh*: 启动单结点训练
- *comm.py*: 实现多结点之间的通信的模块（主要是*allreduce*和分布式采样*sample_data*）
- *comm_test.sh*: 测试comm.py的脚本
- *kill_all.sh*: 杀死参与集合通信的所有进程的脚本（防止出现异常数进程无法关闭）
- *send_to_all_hosts.py*: 将本主机上的所有脚本发给其它结点，确保每个主机都有训练文件
- *train_mn.py*: 多结点的训练脚本（mn为Multiple Node）
- *train_mn.sh*: 启动多结点训练
- *model_sn*: 该目录存储单节点训练的模型
- *model_mn*: 该目录存储单节点训练的模型
- *data*: 该目录存储训练的数据和保存的结果


**3个结点的Allreduce过程**

集群中一共有 $H_0$, $H_1$, $H_2$三台主机，每台主机上的数据分成三份。定义$H_i$上的三块数据为i0, i1, i2，则三个主机上的数据可表示为:
- $H_0$:  00 01 02
- $H_1$:  10 11 12
- $H_2$:  20 21 22


**reduce-scatter过程**

0(2) $\rightarrow$ 1 ：表示 $H_0$ 将它的第2块数据（也就是02）发给 $H_1$ \
0(1) $\leftarrow$ 2 ：表示 $H_0$ 接收来自 $H_2$ 的数据并覆盖或累加到自己的第1块数据上（也就是01）

第一轮：\
0(0) $\rightarrow$ 1  &nbsp;  1(1) $\rightarrow$ 2  &nbsp;  2(2) $\rightarrow$ 0  \
0(2) $\leftarrow$ 2  &nbsp;  1(0) $\leftarrow$ 0  &nbsp;  2(1) $\leftarrow$ 1 
- $H_0$: 00     &nbsp; 01     &nbsp; 02+22
- $H_1$: 00+10  &nbsp; 11     &nbsp; 12
- $H_2$: 20     &nbsp; 11+21  &nbsp; 22

第二轮：\
0(2) $\rightarrow$ 1  &nbsp;  1(0) $\rightarrow$ 2  &nbsp;  2(1) $\rightarrow$ 0  \
0(1) $\leftarrow$ 2  &nbsp;  1(2) $\leftarrow$ 0  &nbsp;  2(0) $\leftarrow$ 1 
- $H_0$: 00       &nbsp; 01+11+21 &nbsp; 02+22
- $H_1$: 00+10    &nbsp; 11       &nbsp; 02+12+22
- $H_2$: 00+10+20 &nbsp; 11+21    &nbsp; 22

**allgather过程**

第一轮：\
0(1) $\rightarrow$ 1  &nbsp;  1(2) $\rightarrow$ 2  &nbsp;  2(0) $\rightarrow$ 0  \
0(0) $\leftarrow$ 2  &nbsp;  1(1) $\leftarrow$ 0  &nbsp;  2(2) $\leftarrow$ 1 
- $H_0$: 00+10+20 &nbsp; 01+11+21 &nbsp; 02+22
- $H_1$: 00+10    &nbsp; 01+11+21 &nbsp; 02+12+22
- $H_2$: 00+10+20 &nbsp; 11+21    &nbsp; 02+12+22

第二轮：\
0(0) $\rightarrow$ 1  &nbsp;  1(1) $\rightarrow$ 2  &nbsp;  2(2) $\rightarrow$ 0 \
0(2) $\leftarrow$ 2  &nbsp;  1(0) $\leftarrow$ 0  &nbsp;  2(1) $\leftarrow$ 1
- $H_0$: 00+10+20 &nbsp; 01+11+21 &nbsp; 02+12+22
- $H_1$: 00+10+20 &nbsp; 01+11+21 &nbsp; 02+12+22
- $H_2$: 00+10+20 &nbsp; 01+11+21 &nbsp; 02+12+22
