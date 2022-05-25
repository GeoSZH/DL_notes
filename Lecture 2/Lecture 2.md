## What to do if my network fails to train?
我们在实际的应用过程中往往不是那么顺利，当遇到困难的时候我们就需要找到问题并提出改进的方案，最终改善我们的结果，这节就讲解的是我们实际应用过程中会遇到的一些问题以及这些问题的解决办法。
### General Guide
![image](https://user-images.githubusercontent.com/88269254/170172975-cb2edc36-3a07-433f-8f7e-1246b680c44d.png)
这里李宏毅老师是拿作业来举例，我们可以代换成我们实际中的例子，那么接下来介绍我们应该怎样一步一步检查：
#### 出了问题我们首先应该检查我们的loss（loss on training data）
如果我们发现我们训练数据的loss就比较大，显然是我们在训练数据上都没有学好，一般有两个可能：
- **Model bias**

   可能的原因就是我们的model太简单了，就像我们之前用linear model，他始终达不到让loss比较小的点，所以这时我们的解决办法就是重新设计我们的model让它更复杂一点。
   
   ![image](https://user-images.githubusercontent.com/88269254/170176930-7313632f-9efe-4c8e-9315-0baeee1913b6.png)
   
- **Optimization**

   当然训练集loss较大时也并不一定完完全全是我们的Model bias责任，也有可能是我们的Optimization环节出了问题，比如我们用Gradient Descent方法找到的时**局部最小值**。
   
那我们怎么判断我们到底哪个环节出了问题呢？

![image](https://user-images.githubusercontent.com/88269254/170177992-c8558e15-e1f2-4511-9de9-a2a11d4aa121.png)

我们可以用不同的模型取跑同一个数据集看看效果究竟是怎么样的，我们想一下，同等情况下针对同一个问题，56-layers的model是要比20-layers更加具有“**弹性的**”，相对来说，56-layers更加不容易出现Model bias的问题，但是实际过程如上图所示，我们在左图中看到，56-layers的测试集误差是要大于20-layers的，为什么会出现这样的现象呢？是overffiting吗？当然不能确定，这时我们再参考一下训练集上的Loss，就会发现，这肯定不是过拟合的问题（为什么？我们增加了36layers，假如说我们前20层与20-layers一模一样，后面全为1也至少应该是20layers的loss，但是右图明显看到我们train loss是高于20-layers的），而是说我们56-layers的模型可能没有找到最小的loss的参数，也就是我们的Optimization issue。

##### 那我们如何确定我们的optimization有没有做好？
如果我们看到一个我们从来没有做过的问题，那我们可以先跑一些比较浅的神经网络，或者甚至于一些linear model，传统的机器学习方法，SVM等等。这些传统的可能较为容易做Optimize，不经常会有Optimize失败的问题，它会找出一组最好的参数。

也就是说我们先找一些比较简单的model去跑一下，然后看一下它们会有什么样的loss，这时我们再用比较deep的模型去跑一下，假如说我们比较深的model没有办法得到的比简单模型更小的loss的话，那就证明我们的optimization有问题。
##### 如何解决optimization的问题？
#### loss较大解决了，那loss较小？
如果我们训练集loss小，那么接下来检查测试集的loss，这里只有一种测试集loss较大情况（因为如果测试集loss也小的话，那训练测试loss都小，已经是一个好model了！）

其实我们在上面也也提到了，如果我们的train loss小，但test loss大的话，那有一种可能是**过拟合**了，那为什么会有over fitting的情况呢？（small loss on training, large loss on testing）

![image](https://user-images.githubusercontent.com/88269254/170193797-d9335be5-3010-4692-8fd4-ade251dc04ee.png)

数学原理可能在lecture4中。

##### 如何缓解overfitting呢？
图中的情况的显而易见的办法是**加入更多数据**，在我们实际处理问题的过程中，可能训练集就是少，那我们应该怎么办？我们可以用**Data augmentation**，通过自己在这个domain的一些先验知识来扩充我们的数据库，比如说在图像识别任务中，我们可以把图像左右翻转，从而多了一倍的训练数据。

还有一种方法就是，也就是说降低我们的模型“**弹性**”（模型复杂度），按照我们对问题的理解调整模型的结构，就像上图中的例子，假如说我们知道它很有可能是一条曲线，那我们一定可以拟合出更好的模型对吧？

其实过拟合的实质是模型的学习能力比较强，把不太一般的特性也学了进来，那么我们针对这种情况就是需要给model做出一些限制。

##### 我们怎么限制model呢？
- **Less parameters， sharing parameters（CNN）**
- **Less features**
- **Early stopping**
- **Regularization**
- **Dropout**

接下来我们可能就陷入了一个困境，如果我们把模型的弹性减小太多（以图中的例子来说就是我们减弱到线性），这样我们就又会出现Model bias的问题。当我们不断增加模型复杂度的时候，train loss会越来越小，但在某一时刻开始，我们的test loss会随之上升，就出现了过拟合的问题。

![image](https://user-images.githubusercontent.com/88269254/170200017-5f55a184-c5b9-491b-8c3c-2dcdb0d33df6.png)

所以我们在挑选模型的时候我们希望能够选到一个比较中庸的模型，在实际的竞赛或者问题中，有的时候并不是测试集成绩最好的模型就一定是最好的，它在最后的验证集可能效果没那么好！

这里李宏毅老师讲了**为什么需要验证集**？（或者说是private testing set）我第一遍没太听懂，第二遍懂了，也就是说哪怕我们是一个简单的model，它的输出是随机猜测的，你训练一次它输出一次，总有那么一次，它猜的全对，输出的结果与test答案完全一致！你能说这个模型是一个非常好的模型吗？当然不能，因为答案是固定的，总有一次能猜中。

#### 怎么解决上述的困境？
##### Cross Validation
![image](https://user-images.githubusercontent.com/88269254/170204306-c3468296-c7c0-4101-8c94-aa61a8b26089.png)

![image](https://user-images.githubusercontent.com/88269254/170205724-bb3e8813-16cb-4239-943c-03f959d6d775.png)

### mismatch
最后我们提到mismatch，我们上面说了test loss比较大的情况可能是overfitting，但没有说肯定是，那是因为还有一种情况是mismatch，它与overfitting最大的区别就是，overfitting可以通过搜集资料克服，但是mismatch是一些由外在影响导致的outlier，比如实例中的，过年那一天的观看人数，因为除夕夜大家都不会点开网站学机器学习，所以这一天就出现了mismatch的情况。

或者更直观的说是，就是我们的测试集的分布与训练集是不同的，就可能会导致我们预测loss很大，mismatch。那我们如何判断它呢？这个就只能通过我们对实际的问题的理解，domain knowledge的帮助了，可以参考HW11.





