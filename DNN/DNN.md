## DNN
### 机器学习的三个步骤
在介绍神经网络之前，李宏毅老师介绍了机器学习的基本概念，其中比较值得注意的是老师介绍了机器学习的三个步骤：
1. **Function with Unknown Parameters**
2. **Define Loss from Training Data**
3. **Optimization**

这里再展开介绍一下其中一些注意的，其中1我们有时可能需要一些先验知识，也就是**base on domain knowledge**，正如李宏毅老师距离说明的本频道观看人数，我们从后台数据得知可能是符合一个线性的规律，所以**y=wx+b**。接下来2中我们要注意**Loss is a fuction of parameters**， **Loss**描述的则是**how godd a set of values is**。我们可以多种方法来计算，比如**MAE**（平均绝对误差），**MSE**（均方误差）等等。当然如果我们在做分类任务的时候，我们可能会用到**cross entropy**。（这里还有一个**error surface**值得注意，感兴趣可以查看）。最后一步就是最优化问题，也就是求**w**<sup>* </sup>，**b**<sup>* </sup>。用的方法则是**Gradient Descent**，这里η是**learning rate**（**hyper parameters**），这也是为什么它代表了收敛速率，因为它和w的更新相关并且掌握着w增大或者减小的步伐，具体见下图：
![image](https://user-images.githubusercontent.com/88269254/170029237-0767dc3b-f59a-4e23-a3a9-c89959ed92cb.png)


再给出一个**Gradient Descent**的实例：

![image](https://user-images.githubusercontent.com/88269254/170030433-5534787b-2dc9-4d85-899b-7fea58ff7fae.png)

### 预测频道观看人数
根据李宏毅老师举得例子，我们最试了**linear models**，但是通过我们对后台人数的观察，我们发现并不是每天的观看人数都在增长，有时在增长有时在下降，因为线性模型的w是固定的，也就是说我们永远无法画出红色的线，这就是**model bias**。
![image](https://user-images.githubusercontent.com/88269254/170038556-d476bd8a-dfd7-43fb-8edd-52de2d35ca96.png)

#### 那么我们怎么实现红色的线呢？
显然我们可以通过**constant** + 一系列蓝色的 **piecewise function** 来实现，我们只需要在每个转折点变换一个piecewise function即可（**图1**），那么更复杂的曲线我们能模拟吗（**图2**）？实际上，而只要增加的点够多后，我们可以用相应数量的piecewise function来模拟任何一个连续曲线（**图3**）！

![image](https://user-images.githubusercontent.com/88269254/170040759-ec04c062-e47d-4eca-8f45-57f5b14717dc.png)
![image](https://user-images.githubusercontent.com/88269254/170040815-a5118b3f-4674-4917-8e56-b0ae9bc19e89.png)
![image](https://user-images.githubusercontent.com/88269254/170040879-d289c9b5-8c31-47a6-b705-08cb8f46e101.png)

#### 那么我们如何去表示这个蓝色的function呢？
**sigmoid！**
![image](https://user-images.githubusercontent.com/88269254/170041498-c7e5db53-33b7-4ef1-9a1d-74c379015750.png)

我们可以看到这个sigmoid函数有三个参数，分别是，w，b，c，那么这三个参数是怎样影响曲线形状的呢？

![image](https://user-images.githubusercontent.com/88269254/170043121-f649de1d-06f7-4baa-bfca-82b7d057d6f7.png)

那么我们的思路就来了，我们就是**要用不同的sigmoid叠加来近似表示不同的piecewise function，进而逼近不同连续曲线**！

### 回到预测观看人数的问题
这样我们就可以改进我们的model，从而写出更加复杂，更加有“**弹性**”的function。但此时我们用的仍是前一天的观看人数，那么如何给我们的model增加特征呢？More Features，我们只需要把sigmoid函数里面的**w**<sub>**i**</sub>**x**<sub>**1**</sub>替换掉即可，我们可以改变j的值换成7天的，换成28天甚至56天的特征。
![image](https://user-images.githubusercontent.com/88269254/170053196-55da1bce-51b6-4180-9346-40dc1677c474.png)

到这里，其实就有点像我们神经网络的计算方式了（也可以说是感知机，但是sigmoid函数我觉得用神经网络更严谨一些），如下图：
![image](https://user-images.githubusercontent.com/88269254/170055155-25a5e395-032c-48fa-b64f-48cef7035f85.png)

### Gradient Descent真正的痛点？

