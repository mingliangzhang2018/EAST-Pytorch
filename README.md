# EAST: An Efficient and Accurate Scene Text Detector

### Introduction
This is a Pytorch re-implementation of [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2).
The features are summarized blow:
+ Only **RBOX** part is implemented.
+ Incidental Scene Text Detection Challenge using only training images from ICDAR 2015 and 2013.
+ Differences from original paper
     + Use Mobilenet-v2 / ResNet-50
     + Use dice loss (optimize IoU of segmentation) rather than balanced cross entropy
     + Use linear learning rate decay rather than staged learning rate decay
+ Every parameter is written in `config.py`, you should change it before you run this project
+ The pre-trained model byprovided achieves ( Mobilenet-v2-**75.01**, ResNet-50-**81.63**) F1-score on ICDAR 2015
+ Speed on 720p (resolution of 1280x720) images:
     + Now
          + Graphic card: GTX 1080 Ti
          + Network fprop: **~50 ms**
          + NMS (Python): **~6ms**
          + Overall: **~16 fps**

Thanks for the code of authors ([@argman](https://github.com/argman)) and ([@songdejia](https://github.com/songdejia))

### Contents
1. [Installation](#installation)
2. [Download](#download)
3. [Train](#train)
4. [Test](#test)
5. [Demo](#demo)
6. [Compute-hmean](#compute-hmean)
7. [Examples](#examples)

### Installation
1. Any version of pytorch version > 0.4.1 should be ok.
2. Other library is instructed in `requirements.txt`.

### Download
1. Models trained on ICDAR 2013 (training set) + ICDAR 2015 (training set): [BaiduYun link](http://pan.baidu.com/s/1jHWDrYQ)
2. Mobilenet-v2 and Resnet V1 50  are provided by : [BaiduYun link](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)

### Train
If you want to train the model, you should change `config.py` parameter. 
1. `train_data_path` is the path of train dataset, put train dataset in this folder. 
2. Depending on your hardware configuration, set parameters of `train_batch_size_per_gpu` `num_workers` `gpu_ids` and `gpu`. 
3. Of course you should specify the pre-training model of backbone_net in `pretrained_basemodel_path` and `pretrained`. 
4. If you want to resume the model of EAST, you should specify the path of `checkpoint` and `resume`. 
5. On the other hand, you could also adjust the setting of other overparameters, such learning rate, weight decay, decay_steps and so on.
6. Then run
```
python train.py
```
*Note: you should change the train and test datasets format same as provided in this project, which the gt text files have same names as image files. In this project, only `.jpg` format image files is accepted. Of course, you can change the code of project.*

### Test
If you want to test the model, you should also change `config.py` parameter.
1. `test_img_path` is the path of test dataset, put test dataset in this folder. `res_img_path` is the path of result which will saved result of images files and txt files.
2. You should also specify the pretrained model in `checkpoint`.
3. Then run
```
python eval.py
```


### Demo
If you only want to test some demos, you downloaded the pre-trained model provided in this project and change `config.py`
1. Put demo images in `.\demo\test_img`, and specify the path of `test_img_path` and `res_img_path`, you will find result in `.\demo\result_img`
2. You should also specify the pretrained model in `checkpoint`.
3. Then run 
```
python eval.py
```

### Compute-hmean
1. put groudtruth of `gt.zip` in `.\dataset\test_compute_hmean`
2. Change parameter of `config.py`, specify the path of `compute_hmean_path`
3. Then run
```
python compute_hmean.py
```
*Note: The result will show in the screen, also record in `.\dataset\test_compute_hmean\log_epoch_hmean.txt` *


### Examples
Here are some test examples on icdar2015, enjoy the beautiful text boxes!

![image_1](demo/result_img/img_1.jpg)
![image_2](demo/result_img/img_2.jpg)

Please let me know if you encounter any issues(my email zhangmingliang2018@ia.ac.cn).

