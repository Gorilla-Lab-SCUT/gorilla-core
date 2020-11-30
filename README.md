[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# gorilla-core

created by [GorillaLab](empty)

## Introduction
A toolbox for learning task for Gorilla-Lab

## Documents
- TODO


## Installation
Run this command in project directory
```sh
python setup.py install(develop)
```


## Demo
**Train**
```sh
python scripts/main.py configs/domain_adaptation/source_only/source_only_resnet34.py --work_dir/$work_dir
```
- this script will search the empty gpu automatically, also you can set `CUDA_VISIBLE_DEVICES` or `--gpu_ids` manually
- the `work_dir` is the directory which you save log and checkpoint (default will be a subdir named `source_only_resnet34` in `work_dirs` in this example)
- if you test `--verbose` will show the details of config, model, data pipeline and hook (data pipeline and hook' show need to complete the comment and `__str__`)

**Test**
```sh
python scripts/main.py configs/domain_adaptation/source_only/source_only_resnet34.py --test
```
- add `--test` will run the test mode



## Reference By
- [open-mmlab/mmcv](https://github.com/open-mmlab/mmcv)
- [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
- [open-mmlab/mmdetection3d](https://github.com/open-mmlab/mmdetection3d)


# TODO
- [x] add utils
- [x] add operations
- [x] modify from mmcv
- [x] run dataset pipeline
- [x] run model demo
- [x] decouple model into model/poster/criterion
- [x] decouple scatter to remove MMDataParallel
- [x] log multiple variables
- [x] simplify the pipeline
- [ ] To be continue

# Trick List
- [x] Warm-up
- [x] Focal loss
- [ ] OHEM
- [ ] S-OHEM
- [ ] GHM
- [ ] Label Smoothing
- [ ] Mix-up
- [ ] KL-Loss
- [ ] loss synchronization
- [ ] automatic BN fusion
- [ ] To be continue


