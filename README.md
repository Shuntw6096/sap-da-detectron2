# A Detectron2 Implementation of [Spatial Attention Pyramid Network for Unsupervised Domain Adaptation](https://arxiv.org/pdf/2003.12979.pdf) (ECCV 2020)

offical implementation: [IntelligentTEAM/ECCV 2020 Domain Adaption](https://isrc.iscas.ac.cn/gitlab/research/domain-adaption)

## Network architecture
<img src='./docs/resnet-sap.png' width=900>  
When target domain feature feed into RPN, RPN loss is not calculated, just generate logit feature for SAP, i.e., target domain annotations are not used.  

## Environment
```
python 3.8.11
pytorch 1.10.0 (conda)
torchvision 0.11.1 (conda)
numpy 1.21.2 (conda)
detectron2 0.6+cu111 (pip)
tensorboard 2.6.0 (pip)
opencv-python (pip)
pycocotools (pip)
```
<details>
<summary>here is full environment</summary>

``` yaml
name: detectron2-cu11
channels:
  - pytorch
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=4.5=1_gnu
  - blas=1.0=mkl
  - bzip2=1.0.8=h7b6447c_0
  - ca-certificates=2021.10.26=h06a4308_2
  - certifi=2021.10.8=py38h06a4308_0
  - cudatoolkit=11.3.1=h2bc3f7f_2
  - ffmpeg=4.3=hf484d3e_0
  - freetype=2.11.0=h70c0345_0
  - giflib=5.2.1=h7b6447c_0
  - gmp=6.2.1=h2531618_2
  - gnutls=3.6.15=he1e5248_0
  - intel-openmp=2021.4.0=h06a4308_3561
  - jpeg=9d=h7f8727e_0
  - lame=3.100=h7b6447c_0
  - lcms2=2.12=h3be6417_0
  - ld_impl_linux-64=2.35.1=h7274673_9
  - libffi=3.3=he6710b0_2
  - libgcc-ng=9.3.0=h5101ec6_17
  - libgomp=9.3.0=h5101ec6_17
  - libiconv=1.15=h63c8f33_5
  - libidn2=2.3.2=h7f8727e_0
  - libpng=1.6.37=hbc83047_0
  - libstdcxx-ng=9.3.0=hd4cf53a_17
  - libtasn1=4.16.0=h27cfd23_0
  - libtiff=4.2.0=h85742a9_0
  - libunistring=0.9.10=h27cfd23_0
  - libuv=1.40.0=h7b6447c_0
  - libwebp=1.2.0=h89dd481_0
  - libwebp-base=1.2.0=h27cfd23_0
  - lz4-c=1.9.3=h295c915_1
  - mkl=2021.4.0=h06a4308_640
  - mkl-service=2.4.0=py38h7f8727e_0
  - mkl_fft=1.3.1=py38hd3c417c_0
  - mkl_random=1.2.2=py38h51133e4_0
  - ncurses=6.3=h7f8727e_2
  - nettle=3.7.3=hbbd107a_1
  - numpy=1.21.2=py38h20f2e39_0
  - numpy-base=1.21.2=py38h79a1101_0
  - olefile=0.46=pyhd3eb1b0_0
  - openh264=2.1.0=hd408876_0
  - openssl=1.1.1l=h7f8727e_0
  - pillow=8.4.0=py38h5aabda8_0
  - pip=21.2.4=py38h06a4308_0
  - python=3.8.12=h12debd9_0
  - pytorch=1.10.0=py3.8_cuda11.3_cudnn8.2.0_0
  - pytorch-mutex=1.0=cuda
  - readline=8.1=h27cfd23_0
  - setuptools=58.0.4=py38h06a4308_0
  - six=1.16.0=pyhd3eb1b0_0
  - sqlite=3.36.0=hc218d9a_0
  - tk=8.6.11=h1ccaba5_0
  - torchvision=0.11.1=py38_cu113
  - typing_extensions=3.10.0.2=pyh06a4308_0
  - wheel=0.37.0=pyhd3eb1b0_1
  - xz=5.2.5=h7b6447c_0
  - zlib=1.2.11=h7b6447c_3
  - zstd=1.4.9=haebb681_0
  - pip:
    - absl-py==1.0.0
    - albumentations==1.1.0
    - antlr4-python3-runtime==4.8
    - appdirs==1.4.4
    - black==21.4b2
    - cachetools==4.2.4
    - charset-normalizer==2.0.8
    - click==8.0.3
    - cloudpickle==2.0.0
    - cycler==0.11.0
    - cython==0.29.24
    - detectron2==0.6+cu111
    - fonttools==4.28.2
    - future==0.18.2
    - fvcore==0.1.5.post20211023
    - google-auth==1.35.0
    - google-auth-oauthlib==0.4.6
    - grpcio==1.42.0
    - hydra-core==1.1.1
    - idna==3.3
    - imageio==2.13.1
    - importlib-metadata==4.8.2
    - importlib-resources==5.4.0
    - iopath==0.1.9
    - jinja2==3.0.3
    - joblib==1.1.0
    - kiwisolver==1.3.2
    - markdown==3.3.6
    - markupsafe==2.0.1
    - matplotlib==3.5.0
    - mypy-extensions==0.4.3
    - networkx==2.6.3
    - oauthlib==3.1.1
    - omegaconf==2.1.1
    - opencv-python==4.5.4.60
    - packaging==21.3
    - pascal-voc-writer==0.1.4
    - pathspec==0.9.0
    - portalocker==2.3.2
    - protobuf==3.19.1
    - pyasn1==0.4.8
    - pyasn1-modules==0.2.8
    - pycocotools==2.0.3
    - pydot==1.4.2
    - pyparsing==3.0.6
    - python-dateutil==2.8.2
    - pywavelets==1.2.0
    - pyyaml==6.0
    - qudida==0.0.4
    - regex==2021.11.10
    - requests==2.26.0
    - requests-oauthlib==1.3.0
    - rsa==4.8
    - scikit-image==0.19.0
    - scikit-learn==1.0.1
    - scipy==1.7.3
    - setuptools-scm==6.3.2
    - tabulate==0.8.9
    - tensorboard==2.7.0
    - tensorboard-data-server==0.6.1
    - tensorboard-plugin-wit==1.8.0
    - termcolor==1.1.0
    - threadpoolctl==3.0.0
    - tifffile==2021.11.2
    - toml==0.10.2
    - tomli==1.2.2
    - tqdm==4.62.3
    - urllib3==1.26.7
    - werkzeug==2.0.2
    - xmltodict==0.12.0
    - yacs==0.1.8
    - zipp==3.6.0
```
</details>

## Data preparation

1. make your dataset format to voc or coco, or [other format](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html) that detectron2 supports
2. register your dataset at [detection/data/register.py](./detection/data/register.py)
3. test set format must be VOC becuase we use VOC metric to evaluate result

```python
# VOC format
dataset_dir = $YOUR_DATASET_ROOT
classes = ('person', 'two-wheels', 'four-wheels') # dataset classes
years = 2007
# to find image list at $YOUR_DATASET_ROOT/ImaegSets/Main/{$split}.txt, only "train", "test", "val", "trainval"
split = 'train'
# call your dataset by this
meta_name = 'itri-taiwan-416_{}'.format(split)
# call register_pascal_voc to register
register_pascal_voc(meta_name, dataset_dir, split, years, classes)
```

## Configuration file explanation
<details>
<summary>Configuration</summary>

``` yaml
# load some basic settings
_BASE_: "./Base-RCNN-C4.yaml"
# dataset settings, souce and target domain dataset, but test set does not have domain setting
DATASETS:
  # domain adaptation trainer's training setting
  SOURCE_DOMAIN:
    TRAIN: ("cityscapes_train",)
  TARGET_DOMAIN:
    TRAIN: ("foggy-cityscapes_train",)
  # default trainer's training setting,
  # when not using domain adaptation, load this training set to train noraml faster-rcnn
  TRAIN: ("cityscapes_train",)
  TEST: ("foggy-cityscapes_val",)
MODEL:
  # code implementation at detection/meta_arch/sap_rcnn.py
  META_ARCHITECTURE: "SAPRCNN"
  BACKBONE:
    # resnet baskbone
    NAME: "build_resnet_backbone"
    # resnet has 5 stages, only freeze stem, same as original SAP setting
    FREEZE_AT: 1
  WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
  KEYPOINT_ON: False
  MASK_ON: False
  # determine whether to use domain adaptation or not, if not, just a normal faster rcnn
  DOMAIN_ADAPTATION_ON: False
  # RPN setting
  PROPOSAL_GENERATOR:
    # code implementation at detection/modeling/rpn.py
    NAME: "SAPRPN"
  ROI_HEADS:
    # use detectron2 resnet default setting
    NAME: "Res5ROIHeads"
    # same as dataset class, it not count background in 
    NUM_CLASSES: 8
    # determine confidence threshold, 
    # boxes are outputed on images while testing if its confidence is above threshold
    SCORE_THRESH_TEST: 0.75
  # Domain adaptation head settings, code implementation at detection/da_heads/sapnet.py
  DA_HEADS:
    # input, feature comes from backbone
    IN_FEATURE: "res4"
    # IN_FEATURE channel
    IN_CHANNELS: 1024
    # how many different size anchors in image for anchor generator, len(anchor_size) * len(aspect_ratio)
    NUM_ANCHOR_IN_IMG: 15
    EMBEDDING_KERNEL_SIZE: 3
    EMBEDDING_NORM: True
    EMBEDDING_DROPOUT: True
    # loss function, only supports cross entropy
    FUNC_NAME: "cross_entropy"
    # spatial pyramid pooling function, supports max and avg
    POOL_TYPE: 'avg'
    # adversarial loss weight, constant during training
    LOSS_WEIGHT: 1.0
    # spatial pyramid pooling setting
    WINDOW_STRIDES: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    WINDOW_SIZES: [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35, 37]
INPUT:
  # data augmentation setting, resize short edge
  MIN_SIZE_TRAIN: (800, 832, 864, 864, 896, 928, 960, 992, 1024)
  MIN_SIZE_TRAIN_SAMPLING: "choice"
  MAX_SIZE_TRAIN: 2048
  # not to resize input during testing
  MIN_SIZE_TEST: 1024
  MAX_SIZE_TEST: 2048
# optimizer setting, SGD is used in baseline training, Adam is used in SAP training
SOLVER:
  IMS_PER_BATCH: 1 # batch size
  # learning rate decay step
  STEPS: (70000, 80000)
  # learning rate
  BASE_LR: 0.00001
  MAX_ITER: 90000
  CHECKPOINT_PERIOD: 5000
TEST:
    # determine how many steps to run inference on test set to get metric(mAP), 0 is not to run
  EVAL_PERIOD: 5000
# determine how many steps to get image record(eg., predicted proposal generated by rpn) during traing,
# smaller to make tfevents file larger 
VIS_PERIOD: 5000
```
</details>

* [baseline_R_50_C4_1x-city2foggy.yaml](./configs/baseline_R_50_C4_1x-city2foggy.yaml) is a normal fatser rcnn configuration file  
* [sap_R_50_C4_1x-city2foggy.yaml](./configs/sap_R_50_C4_1x-city2foggy.yaml) is faster rcnn with SAP configuration file

### Usages
1. Train a model without SAP (normal faster rcnn) using source domain data (baseline)
2. Train whole model using baseline model weight
* train a model
``` bash
python tools/train_net.py --config-file $CONFIG_FILE_PATH --num-gpus 1
```
* test the model
``` bash
# update MODEL.WEIGHTS by command line
python tools/train_net.py --config-file $CONFIG_FILE_PATH --num-gpus 1 --eval-only MODEL.WEIGHTS $MODEL_WEIGHT_PATH
```
* predict boxes on the test set
``` bash
python tools/train_net.py --config-file $CONFIG_FILE_PATH --num-gpus 1 --test-images MODEL.WEIGHTS $MODEL_WEIGHT_PATH  MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.75
```
## Experiments
* Cityscapes -> Foggy Cityscapes

| Setting | Backbone | person | rider | car | truck | bus | train | motorcycle | bicycle | mAP |
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| baseline, w/o DA, paper | vgg16   | 24.1 | 33.1 | 34.3 | 4.1 | 22.3 | 3.0 | 15.3 | 26.5 | 20.3|
| SAP, w/ DA, paper | vgg16         | 40.8 | 46.7 | 59.8 | 24.3 | 46.8 | 37.5 | 30.4 | 40.7 | 40.9 |
| baseline, w/o DA, ours | resnet50 | 33.65 | 38.99 | 39.98 | 22.42 | 22.61 | 9.091 | 26.93 | 37.42 | 28.83 |
| SAP, w/ DA, ours | resnet50       | 47.02 | 52.82 | 57.64 | 29.36 | 43.61 | 26.12 | 31.98 | 48.75 | 41.7 |

* GTA5 -> Cityscapes

| Setting | Backbone | AP on car |
|:-------------:|:-------------:|:-------------:|
| baseline, w/o DA, paper | vgg16   | 34.6 |
| SAP, w/ DA, paper | vgg16         | 44.9 |
| baseline, w/o DA, ours | resnet50 | 38.24 |
| SAP, w/ DA, ours | resnet50       | 44.8 |


