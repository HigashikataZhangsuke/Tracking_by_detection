# Tracking_by_detection
A Python3-based system use yolov3/KCF&amp;DSST/KF to detect,tracking and estimate the target（people and cars） location.
这是我的本科毕设项目，主要通过YOLOv3识别，利用KCF-DSST-APCE完成抗遮挡的尺度变化跟踪，并用卡尔曼滤波估计位置。

# Test environment

Intel Core i5-8300H CPU@2.30GHz

Nvidia GTX 1050Ti

Ubuntu 18.04LTS

Python3

OPENCV3.4.4

# Requirements

CUDA9.0

CUDNN7

Python3

OPENCV

# Code structure

yolo.py:detect 

fhog.py:give fhog feature

tracker.py:track

run.py:basic framework

run2.py:add more visualization work for some video

# Cites and Others' codes' citation
@article{redmon2018yolov3,
  title={Yolov3: An incremental improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal={arXiv preprint arXiv:1804.02767},
  year={2018}
}

@article{henriques2014high,
  title={High-speed tracking with kernelized correlation filters},
  author={Henriques, Jo{\~a}o F and Caseiro, Rui and Martins, Pedro and Batista, Jorge},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={37},
  number={3},
  pages={583--596},
  year={2014},
  publisher={IEEE}
}

@inproceedings{danelljan2014accurate,
  title={Accurate scale estimation for robust visual tracking},
  author={Danelljan, Martin and H{\"a}ger, Gustav and Khan, Fahad and Felsberg, Michael},
  booktitle={British Machine Vision Conference, Nottingham, September 1-5, 2014},
  year={2014},
  organization={BMVA Press}
}

@inproceedings{wang2017large,
  title={Large margin object tracking with circulant feature maps},
  author={Wang, Mengmeng and Liu, Yong and Huang, Zeyi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4021--4029},
  year={2017}
}

The YOLOv3 part is based on yehengcheng's work @ https://github.com/yehengchen/Object-Detection-and-Tracking

The KCF and DSST part is based on ryanfwy's work @ https://github.com/ryanfwy/KCF-DSST-py

# Install and use
## First you should clone this repository:

> git clone https://github.com/sjtuzyz/Tracking_by_detection.git

## Then change some part of code to slove your task

### Go to yolo.py line 122 

> if  predicted_class != 'person':

The class should be changed to person or car or whatever you want.About the yolo network's training and convert,please check yehengcheng's work.

### Go to tracker.py line 463

```python
        if (APCE>0.12):#25 oc1.mp4 0.02 oc2.mp4
            x = self.getFeatures(image, 0, 1.0)
            self.train(x, self.interp_factor)
        #x = self.getFeatures(image, 0, 1.0)
        #self.train(x, self.interp_factor)
```

If you want to use APCE to let the system more robust to occlusions ,just change 0.12 to a threshold of your test video, which will be print in the terminal.

### Go to run.py(or run2.py and find where to change by yourself)line 50-78 and 109-136

I use similar geometric relations to calculate the distance,which means you need to get the Intrinsic Matrix K by Matlab or any other code and target's height or width first.Then just change the parameters to yours.

### Go to run.py(or run2.py and find where to change by yourself)line 205

> c = cv2.waitKey(1) & 0xFF

change the waitkey to get the frame played slower.
