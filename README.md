# Real-time multi-object, segmentation and pose tracking using Yolov8 | Yolo-NAS | YOLOX with DeepOCSORT and LightMBN


<div align="center">
  <p>
  <img src="boxmot/strongsort/results/track_all_seg_1280_025conf.gif" width="400"/>
  </p>
  <br>
  <div>
  <a href="https://github.com/mikel-brostrom/yolov8_tracking/actions/workflows/ci.yml"><img src="https://github.com/mikel-brostrom/yolov8_tracking/actions/workflows/ci.yml/badge.svg" alt="CI CPU testing"></a>
  <a href="https://pepy.tech/project/boxmot"><img src="https://static.pepy.tech/personalized-badge/boxmot?period=month&units=international_system&left_color=grey&right_color=orange&left_text=Downloads"></a>
  <br>  
  <a href="https://colab.research.google.com/drive/18nIqkBr68TkK8dHdarxTco6svHUJGggY?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
<a href="https://doi.org/10.5281/zenodo.7629840"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7629840.svg" alt="DOI"></a>
  </div>
</div>


## Introduction

This repo contains a collections of state-of-the-art multi-object trackers. Some of them are based on motion only, others on motion + appearance description. For the latter, state-of-the-art ReID model are downloaded automatically as well. Supported ones at the moment are: [DeepOCSORT](https://arxiv.org/abs/2302.11813) [LightMBN](https://arxiv.org/pdf/2101.10774.pdf), [BoTSORT](https://arxiv.org/abs/2206.14651) [LightMBN](https://github.com/jixunbo/LightMBN)[](https://arxiv.org/pdf/2101.10774.pdf), [StrongSORT](https://github.com/dyhBUPT/StrongSORT)[](https://arxiv.org/abs/2202.13514) [LightMBN](https://github.com/jixunbo/LightMBN)[](https://arxiv.org/pdf/2101.10774.pdf), [OCSORT](https://github.com/noahcao/OC_SORT)[](https://arxiv.org/abs/2203.14360) and [ByteTrack](https://github.com/ifzhang/ByteTrack)[](https://arxiv.org/abs/2110.06864).

We provide examples on how to use this package together with popular object detection models. Right now [Yolov8](https://github.com/ultralytics), [Yolo-NAS](https://github.com/Deci-AI/super-gradients) and YOLOX are available.


  

## Contact 

For Yolov8 tracking bugs and feature requests please visit [GitHub Issues](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/issues). 
For business inquiries or professional support requests please send an email to: yolov5.deepsort.pytorch@gmail.com
