# Container detection and container number OCR
Container detection and container number OCR is a specific project requirement, and using [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to verify feasibility is one of the quickest and simplest way.

# 说明
## 写在前面的话
>&ensp;&ensp;&ensp;&ensp;两年多之前我在“ex公司”的时候，有一个明确的项目需求是集装箱识别并计数，然后通过集装箱号OCR识别记录每一个集装箱号，然后与其余业务系统的数据进行交换，以实现特定的需求。正好Tensorflow Object Detection API 发布了，就放弃了YOLO或者SSD的选项，考虑用TF实现Demo做POC验证了。<br>
&ensp;&ensp;&ensp;&ensp;之前也并未接触过Deep Learning相关的事情，为了验证这个需求可以实现并满足交付要求，从入门到了解到选择到学习到准备图片并标注到完成基本的Python Demo验证一个人前后差不多用了2个月时间（那时候用的还是12年中的MacBook Air），验证完成后，交给了C++和C#的小伙伴们去实现。在这个过程中感受到了Deep Learning的魅力与强大以及对于未来的无限可能，就带着我的研发小伙伴们走向这条路前进了，人脸、语音这些都尝试了，也都嵌入了已有都产品中，也不会因为只有第三方算法而各种踩坑了。<br>
&ensp;&ensp;&ensp;&ensp;最近重装了Mac，重做一遍这个demo验证下Inteld对于CPU的DL支持有多大改善。
### 主要内容
#### 1、明确要具体解决的问题
&ensp;&ensp;&ensp;&ensp;这个事情的需求有两个关键点：1. 如何准确识别集装箱，识别后如何去重才能保证“计数”的准确；2. 如何在单摄像头且非约束的场景下做集装箱号的OCR。已有的解决方案，集装箱识别——没有；集装箱号OCR——有，但是都是约束场景下的识别，即集装箱与摄像头的位置关系有明确限制且很有可能使用多达3个摄像头做联合判断后输出结果，与实际场景不符，等于没有。<br>
&ensp;&ensp;&ensp;&ensp;所以这个需求能否实现，关键就在于单摄像头非约束场景下的detection与OCR了。
#### 2、为什么不用SSD与YOLO
&ensp;&ensp;&ensp;&ensp;这两个都很快，识别准确率也足够，之前确实考虑选其中之一来实现，但是想明白了这件事怎么做，也准备好了几千张图片准备验证的时候 Object Detection API 发布了，DL这种对未来影响深远的事情，选择一个最有可能成为大生态环境的平台就很重要了，Google虽然之前有“说关就关”的“前科”但是在我的理解不论软件还是硬件层面，DL的inference能力就是未来的OS之上最重要的事情了，甚至可以理解为就是OS的一部分（比如SoC），所以，这事情Google任性可能性微乎其微，从这两年发展来看，Google对TF持续投入，而且做为“大厂”的标志之一似乎就是你有没有开源的DL相关工作。
#### 3、当时不担心搞不定么？
&ensp;&ensp;&ensp;&ensp;担心，真的担心搞不定，所以连一块1080Ti的采购都没敢提，前期工作全部用我的12年老MBA完成的。就是由于真的怕搞不定，所以我没有玩弄所谓的“权术”。我是公司的研发总，所以理论上，我把这个Demo的预研工作交出去，找两个人去搞定，搞定了我工作依然完成了，搞不定我还有“替死鬼”。我也不是没事干，产品、架构、评审、项目、质量保证、销售以及没完没了的各种会议，但是这个事情真没底，不知道能不能搞定，安排工作简单，但是安排自己都没把握的工作出去虽然是一种“管理智慧”，但是不是本人做事、为人风格。于是就白天干工作职责之内的事情，下班后再来做这件事，能通宵就通宵，周末？周末是不可能有的。所以说起来2个月时间，其实是2个月的业余时间。<br>
&ensp;&ensp;&ensp;&ensp;DL2018年持续火爆，想上车的同学越来越多，在此如果你机会看到这段文字，那么我可以负责任的告诉你：只要你全情投入，肯定可以上车并且走的很远。
## 基本pipeline
### 定义目标
……待补充

### 采集数据、标注
……待补充

### 选择模型、训练
……待补充

### pipeline的demo实现
……待补充

# 用法

……待补充


# 相关资源
……待补充

# 参考

[Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

[Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
