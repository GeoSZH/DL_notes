## What to do if my network fails to train?
我们在实际的应用过程中往往不是那么顺利，当遇到困难的时候我们就需要找到问题并提出改进的方案，最终改善我们的结果，这节就讲解的是我们实际应用过程中会遇到的一些问题以及这些问题的解决办法。
### General Guide
![image](https://user-images.githubusercontent.com/88269254/170172975-cb2edc36-3a07-433f-8f7e-1246b680c44d.png)
这里李宏毅老师是拿作业来举例，我们可以代换成我们实际中的例子，那么接下来介绍我们应该怎样一步一步检查：
#### 1. 出了问题我们首先应该检查我们的loss（loss on training data）
如果我们发现我们训练数据的loss就比较大，显然是我们在训练数据上都没有学好，一般有两个可能：
- **Model bias**

   可能的原因就是我们的model太简单了，就像我们之前用linear model，他始终达不到让loss比较小的点，所以这时我们的解决办法就是重新设计我们的model让它更复杂一点。
   
   ![image](https://user-images.githubusercontent.com/88269254/170176930-7313632f-9efe-4c8e-9315-0baeee1913b6.png)
   
- **Optimization**

   当然训练集loss较大时也并不一定完完全全是我们的Model bias责任，也有可能是我们的Optimization环节出了问题，比如我们用Gradient Descent方法找到的时**局部最小值**。
   
那我们怎么判断我们到底哪个环节出了问题呢？

![image](https://user-images.githubusercontent.com/88269254/170177992-c8558e15-e1f2-4511-9de9-a2a11d4aa121.png)

我们可以用不同的模型取跑同一个数据集看看效果究竟是怎么样的，如上图所示，我们在左图中看到，

