# Container detection and container number OCR using Tensorflow Object Detection API and Tesseract

Container detection and container number OCR is a specific project requirement, using [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [Tesseract](https://github.com/tesseract-ocr/tesseract) to verify feasibility is one of the quickest and simplest ways.

>两年多之前我在“ex公司”的时候，有一个明确的项目需求是集装箱识别并计数，然后通过集装箱号OCR识别记录每一个集装箱号，然后与其余业务系统的数据进行交换，以实现特定的需求。正好Tensorflow Object Detection API 发布了，就放弃了YOLO或者SSD的选项，考虑用TF实现Demo做POC验证了。具体需求实现的思考与pipeline构想思考参见这篇文章：[Container detection and container number OCR](https://lonelygo.github.io/2019-01-20-container-detection/) 。  

## 用法

### Tensorflow Object Detection API 安装

具体安装参考官方[说明](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)。  

### 环境与依赖

本人使用的环境是：macOS 10.14.2，python 3.6.8，TF 1.12
除了Tensorflow Object Detection API 安装必备的依赖外，还需要以下依赖：
tesseract
pytesseract
具体安装及用途，请自行Google。
`visualization_utils.py`中:

``` python
import matplotlib; matplotlib.use('Agg')
```

Agg在我的环境下用不了，也懒得折腾，所以把这句改了。

### 数据集准备

参考PascalVOC的数据集格式，使用[LabelImg](https://github.com/tzutalin/labelImg)进行标注。  
标注完成后可以使用`generate_voc_datasets.py`按你的想法分割数据集为：train 、val 与 test这个三个data set。
分割为三个data set后，可以使用`create_pascal_tf_record.py`转换为TF record格式data set文件供TF使用（此文件官方提供，在`/object_detection/dataset_tools/`）。  
有关数据准备的内容，可以参考这里的[说明](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)。

### 训练

参考[官方说明-本地](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md)使用官方代码库中的`model_main.py`在本地训练(以前是train 和 val 分别提供了两个版本，目前版本用这一个文件就可以了。)。
参考[官方说明——Google Cloud ML Engine](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_cloud.md)在Google Cloud ML Engine上使用TPU训练，资费说明在[这里](https://cloud.google.com/ml-engine/docs/tensorflow/pricing?hl=zh-CN)，可以选择“竞争”模式使用，会便宜很多。

### 验证

可以使用官方代码中的`object_detection_tutorial.ipynb`做快速验证尝试。本repo中的`detection_var_image.py`也主要参考这个ipynp实现的。
以下几个位置需要根据你自己的实际情况来修改：

``` python

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'container_label_map.pbtxt')

TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 4)]

lang = 'cont41'

```

其中`lang = 'cont41'`中的`cont41`是trsseract使用的lang文件的名字，如果你还没有来得及自己训练lang文件，可以把`lang_use = 'eng+'+lang+'+letsgodigital+snum+eng_f'`中的其余内容都删了，仅保留`eng`，使用tesseract安装默认带的lang文件进行识别。  
返回的`image_label`为一个嵌套列表，会是这个样子：  

``` python

[{'image1': [{'lable': 'container_number', 'actual': '100%', 'cont_num': 'TCLU § 148575 3\n45G1', 'image_corp_name': 'image1_1_container_number'}]}, {'image2': [{'lable': 'container_number', 'actual': '99%', 'cont_num': 'TRNU816699 4 |\n45G1', 'image_corp_name': 'image2_1_container_number'}, {'lable': 'container_number', 'actual': '99%', 'cont_num': 'TCNU89092898\n4561', 'image_corp_name': 'image2_2_container_number'}, {'lable': 'container_number', 'actual': '99%', 'cont_num': 'MSKUY 86801264\n4561', 'image_corp_name': 'image2_3_container_number'}]}]

```

每个索引对应一个字典，字典的：  
`key`为输入的图片名称；  
`value`为一个列表，列表的索引对应的是由4个key构成的字典，分别是标签、置信度、OCR的结果以及输出的裁剪后的集装箱号图片的名称，索引数量则代表了在图片中找到的集装箱号。

主要是考虑如果再用flask做个Web，可以直接用flask简单做个服务端，把检测的结果JSON串一次性抛出来，Demo环节没必要再单独折腾TensorFlow Serving部署一个后端。

对于每张输入的图片，除了上述JOSN输出外，还输出：  
绘制了Bounding box 与 label 的图片；  
集装箱号位置的裁剪图片（有几个裁几个），以及使用openCV做了预处理后丢入tesseract之前的图片。通过对比图片与OCR结果，可以给我们调整图片预处理的思路与参数。

#### Demo

`image`文件夹下有5张测试图片，测试结果在`cont_num.txt`中，部分如下：

| 图片名 | OCR结果 | 实际 |
|:------:|:------:|:----:|
| image1_1_container_number_100% | TCLU § 148575 3 45G1 | TCLU 148575 3 45G1 |
|image2_1_container_number_99% | TRNU816699 4 \| 45G1 | TRLU 818699 0 45G1 |
| image2_2_container_number_99% | TCNU89092898 4561 | TCNU 869248 8 45G1 |
| image2_3_container_number_99% | MSKUY 86801264 4561 | MSKU 868012 6 4561 |
| image3_1_container_number_99% | x L BOUL 871489 7 \| 221 | BMOU 871489 7 22R1 |
| image3_2_container_number_99% | FCIU [599867 (0 22G1 | FCIU 599887 0 22G1 |

可以看到，OCR的整体准确率并不高，可以说，与我在[Container detection and container number OCR](https://lonelygo.github.io/2019-01-20-container-detection/)中预估的准确率不超过8成是匹配的（现在看肯定是事后诸葛亮，但在当时下决心做验证的时候是这么一个真实预测）。这个准确率并不是没有提高可能的，实际上在以下几个方面可以继续做一些工作进行尝试：

- 因为Tesseract训练用的图片质量大多和`image1.jpg`接近，所以需要调整训练集的图片质量，使其比较符合工程场景图像质量；
- 工程场景下，尽量保证图像质量，并且通过工程现场使用，收集图片；
- 图片收集足够数量后，OCR引擎转变为深度学习版本的；
- 改善OCR之前的图像预处理策略，事实上，我在换了其他的预处理策略后，结果是可以优于上述表现的。

其中，`image1.jpg`输出图片分别如下：
![Bounding box](https://ws1.sinaimg.cn/large/55fc1144gy1fzkay5dqltj20qo0zk42p.jpg)

![original](https://ws1.sinaimg.cn/large/55fc1144gy1fzkaz64bcxj209t03ydfr.jpg)

![gray](https://ws1.sinaimg.cn/large/55fc1144gy1fzkazp3xcij209t03ywel.jpg)

#### To Do

- [ ] 增加使用视频流检测的Demo版本
- [ ] 用flask增加一个简单的Web上传与显示结果的页面

#### 参考

[Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

[Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
