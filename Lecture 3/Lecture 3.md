## Lecture3: Images as input
### Convolutional Neural Network（CNN）
#### Image Classification
在这部分我们假设图片大小都是一致的，常见的处理方式是首先将要处理的图片ReScale至同样的大小。大概的流程如下：

![img.png](figure/img.png)

**那么图片是怎么作为模型的输入呢？**

其实一张图片是一个三维的Tensor，一个维度代表长另一个维度代表宽，最后一个维度表示channel数目，而channel数目的解释，我们可以把每个pixel看成是由RGB组成，如图所以就变成了一个100* 100* 3，我们需要输入的是一个向量，那么怎么处理呢？可以把它拉直排至一排（列），就变成了如图所示的一个输入：

![img_1.png](figure/img_1.png)

而矩阵中的每一个数字可以代表图片中对应位置某个颜色的强度。假如我们以100* 100 * 3这样输入，并且假设下一层的neural有1000个，那我们需要的参数w就足足有3*10^7个（如下图），这么多的参数就可能会带来问题：比如参数较多我们的模型弹性较大，然后我们就会比较容易的出现overfitting的问题，所以全连接的神经网络在这里其实并不适用，我们应该怎么办呢？

![img.png](figure/img_2.png)

#### Observation 1
考虑到图像识别的特性，其实我们并不需要全连接的神经网络，为什么？举个例子，我们是怎么通过眼睛来识别物体的呢？我们是通过锁定一些局部特征来确定我们对某个物体的认知的，而神经网络同样可以把这些特征作为要识别的pattern来识别，而且这样也更加精确，如下：

![img.png](figure/img_25.png)

#### Simplification 1
我们可以定义一个Receptive field，然后让每一组神经元都负责一个Receptive field，找出其中的pattern，这样就可以大大解决我们之前遇到的问题，但这里可能会有一些小问题，比如，右上角的三个问题，肯定都是可以的，这都是可以自己去根据对问题的把握程度来调节的，而且神经元的Receptive field可以相互重叠，因为有的时候可能最初的划分不一定能够找出我们需要的pattern：（这里可能有问题就是Receptive field一定是相连的吗？其实不一定，可以设置成不相连的，但是根据我们对现实问题的理解，pattern出现在一个相连区域的情况占据了绝对的优势）

![img_1.png](figure/img_26.png)

**Simplification 1 - Typical Setting**
一般来说我们图像识别时都是看所有的channels的，但是并不是一定需要这么做，有的时候可能只看某个channel，但这种情况很少，所以一般是默认看所有channels的。于是一般讲kernel size（一般设置为3* 3，7 *7，9 *9就属于比较大的了 ）的时候不讲channel数目，然后通过控制步长来stride遍历图片，一般stride是比较小的，为的就是尽可能多的捕获pattern，而有的时候我们可能会超出范围，这时候我们需要padding，有很多不同的补值方式，一般都是补0，如下图：

![img_2.png](figure/img_27.png)

#### Observation 2
因为图像识别的关系，我们会应对各式各样的图片，那么假如说我们是对鸟嘴进行识别，如下图所示，鸟嘴既可能出现在左上角，同样有可能出现在中间，那么我们难道需要对每一个我们的Receptive field都进行鸟嘴探测吗？如果这样的话，那就又大大增加了参数的数量，回到了我们最初的困境：

![img_3.png](figure/img_3.png)

#### Simplification 2
这里的解决办法就是parameter sharing，共享参数可以很好的解决这个问题，因为哪怕是我们不同的Receptive field共享了参数，但是由于其关注的区域的值不相同，所以我们得到的结果也是不完全一样的。

![img_4.png](figure/img_4.png)

我们之前说了每一个Receptive field都对应有一组神经元（比如32个，64个等等），而不同的Receptive field共享参数就是通过共享其对应的神经元的参数来实现sharing的，如下图：

![img_5.png](figure/img_5.png)

#### Benefit Convolutional Layer
Receptive Field + Parameter Sharing = Convolutional Layer，这其实也是我们的简化过程：

![img_6.png](figure/img_6.png)

#### 具体过程中的filter
如下图，我们可以看到一个图片输入（彩色图片channel=3，黑白图片channel=1），然后进入卷积层会经过一堆filter：

![img_7.png](figure/img_7.png)

这些filter是通过如下方式计算的（这里假设的channel=1，如果channel=3的话我们的filter也是随之变化），filter里的参数这里假设了已知，实际过程中我们是通过Gradient Descent来计算的：

![img_8.png](figure/img_8.png)

然后我们就通过相乘相加然后一个一个stride来计算，最后得出右下角的Filter1处理后的结果，我们可以看出来Filter1是建设对角线上的特征，然后以此类推Filter2，最后组成了一个Feature Map：

![img_9.png](figure/img_9.png)

![img_10.png](figure/img_10.png)

而我们在实际过程中可能并不一定只有一个卷积层，可能会有多个，我们假设第一个卷积层64filters，这样我们得到的是一个有着64channels的一个新的tensor，然后我们再用这个数据来通过下一个卷积层：

![img_11.png](figure/img_11.png)

这里还有一个小问题就是，假如我们一致设置一个3* 3的这样大小的一个Receptive field，会不会捕捉不到比较大的patterns？当然不会，我们可以思考一下，在经过第一个卷积层的时候，我们就已经把原来的图片的长和宽变小了，如图所示，当我们第二次卷积仍然使用3* 3时，我们这是对应到原来的已经变成了5* 5的一个Receptive field。

![img_12.png](figure/img_12.png)

在实际的过程中我们就是通过对应的参数和Receptive field里面的数值进行对应从而计算convolution之后的值，这里其实简化了bias，在实际过程中相乘之后同样有bias项需要加上：

![img_13.png](figure/img_13.png)

我们讲的分配着不同区域的神经元共享参数时通过filter共享这一下应该就明白了，我们每个filter都是扫过整张图片的：

![img_14.png](figure/img_14.png)

最终这就是我们上面提到的两个方面把，在具体过程中的filter上面我们介绍了神经元的思路，可能不太好理解，但是我们看完filter之后应该就能明白上面的神经元视角的介绍了：

![img_15.png](figure/img_15.png)

#### Observation 3
下面我们将介绍Pooling，从下图可以看出来Pooling是把图片缩小了，类似于下采样：

![img_16.png](figure/img_16.png)

Pooling的方法其实有比较多种，有最大值最小值的还有一些其他方法的，这里介绍的是Max Pooling，简而言之就是我们在一个区域里取一个最大值来代表它：

![img_17.png](figure/img_17.png)

在实际过程中我们也是经常运用Convolution和Pooling的组合来提取信息，要注意的是，Pooling过程中是不改变channels数目的，所以channel数目还是由卷积层filters数量来决定，比方下图，我们Pooling就得到了一个较小的tensor：

![img_18.png](figure/img_18.png)

但近些年来由于计算性能的飞速发展，有些已经开始不做Pooling了，因为Pooling本质还是减少计算量，而现在有部分时候需要我们进行一些细节的查找的话，我们用Pooling反而会忽略到一些很小的东西。

#### The whole CNN
我们常见的CNN架构就如下，首先是我们前面介绍的卷积层和池化层，之后便是flatten，因为我们在卷积的时候会产生很多的channel，这里就是把tensor拉直成为一个向量，这样才能够进入到全连接层，然后计算，最后可能还需要经过softmax，最后完成图像识别：

![img_19.png](figure/img_19.png)

#### Application
让人意想不到的是著名的Alpha Go就是用的CNN，我们假想一下下围棋的这么一个场景，似乎确实可以用Neural Network来解决，我们输入的可能是一个19* 19的向量，然后我们得到一个19* 19的分类告诉我们下一步应该下在哪，但是Alpha Go其实用的是CNN结构：

![img_20.png](figure/img_20.png)

难道说用CNN只是为了识别棋盘吗，当然不是，其实是因为它与图像识别有一些异曲同工吧，我们可以把期盼看作是一个19* 19的图片，然后Alpha Go的设计师给出了48个Channel，这个Channel数目可能有更深层次的原因，原文章中也没有过多解释。

为什么说和图像识别有相似呢，我们思考一个如下的问题，我们其实可以把棋盘分作是不同的小区域（Alpha Go用的是5* 5），如图所示我们看这个区域，Alpha Go就更能明白接下来应该怎么做，而且，这样类似的区域可能出现在棋盘当中的任何区域，这就仿佛我们把它看作是一个patterns，然后我们再整张图片中寻找patterns：

![img_21.png](figure/img_21.png)

这里可能有质疑就会说，假设上面说的都对，那么Pooling层岂不是对围棋的巨大削弱，查阅Alpha Go的结构便可知道，其实它并没有Pooling层：

![img_22.png](figure/img_22.png)

这也给我们了一个启发，我们可以把CNN用在不同的领域，但是应用的时候我们要根据具体的问题来调整我们的结构，不能单一的拿处理image的结构来套用其他问题，这些model都有优点也有缺点，我们要在实际问题中把model的优点发挥出来契合问题，并且要把model对于问题来说的致命缺点想办法干掉：

![img_23.png](figure/img_23.png)

#### Data augmentation
最后为什么要进行Data augmentation？

因为在实际过程中给一个图片可能CNN学习过后，你把图片的局部放大CNN可能就不认识它了（这是真的！）所以我们就是要通过放大，旋转来让CNN捕捉到更多细节，泛化能力更强：

![img_24.png](figure/img_24.png)

123