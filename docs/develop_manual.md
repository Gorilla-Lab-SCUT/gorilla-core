# gorilla2d/3d 开发指南

我们建立 `gorilla2d/3d` 的目的在于更好地对实验室的代码进行管理，同时，我们的代码库处于比较初始的阶段，为了更好地在辅助同学们开发的同时建立我们的代码库，我们希望同学们在平时开发的过程中尽可能地利用我们代码库已有的东西，同时也尽可能地向代码库贡献代码。
如何利用 `gorilla-core` 辅助开发在 `transfer_manual` 中已经进行了说明，接下来该手册的目的在于如何更好地进行开发。

---

## \_\_init\_\_.py
在 Python 库的开发中 `__init__.py` 的定义是不可缺少的，接下来会简要说明一下 `__init__.py` 所起到的作用，以 `gorilla3d` 库为例，其结构为：
```sh
gorilla3d
├── datasets
├── evaluation
├── losses
├── nn
├── ops
├── post
├── utils
├── version.py
└── __init__.py
```
可以看到该目录下有一个 `__init__.py`，实际上该目录下的文件夹中都有 `__init__.py`，以 `datasets` 为例：
```sh
datasets/
├── __init__.py
├── scannetv2
│   ├── __init__.py
│   └── scannetv2_inst.py
└── utils
    ├── __init__.py
    ├── pc_aug.py
    └── transforms.py
```
那么 `__init__.py` 起到的作用是什么，简单来说就是起到了定义目录层级结构的功能。依旧以该库为例，假如我在 `gorilla3d/datasets/scannetv2/scannetv2_inst.py` 定义了 `ScanNetV2Inst` 类，那么我们可以通过以下形式引用：
```python
# 调用方式一
import gorilla3d.datasets.scannetv2.scannetv2_inst as scannetv2_inst
dataset_caller = scannetv2_inst.ScanNetV2Inst
# 调用方式二
from gorilla3d.datasets.scannetv2.scannetv2_inst import ScanNetV2Inst
dataset_caller = ScanNetV2Inst
```
而现在在 `__init__.py` 的帮助下，我们可以直接实现：
```python
import gorilla3d
dataset_caller = gorilla3d.ScanNetV2Inst # 直接调用
dataset_caller = gorilla3d.datasets .ScanNetV2Inst # 分级调用，防止弄混

from gorilla3d import ScanNetV2Inst
dataset_caller = ScanNetV2Inst
```
可以看到极大地方便了使用代码库的同学的调用。

下面说一下 `__init__.py` 的工作原理：以 `gorilla3d` 库为例，当我们运行 `import gorilla3d` 时，它会对该目录进行搜索，如果检索到 `__init__.py` 则自动运行，`gorilla3d/__init__.py` 内容如下：
```python
# Copyright (c) Gorilla-Lab. All rights reserved.
from .version import __version__
from .ops import *
from .utils import *
from .nn import *
from .losses import *
from .evaluation import *
from .datasets import *
from .post import *
```
可以看到 `__init__.py` 起到的功能非常简单，从 `version.py` 中获取版本号以及从各子目录中获取相应的声明，那么我们以 `datasets` 为例，观看其子文件夹中的 `__init__.py` ，了解是怎么实现直接调用 `ScanNetV2Inst` 的：
```python
# datasets/__init__py 内容
from .scannetv2 import ScanNetV2Inst
__all__ = [k for k in globals().keys() if not k.startswith("_")]

# datasets/scannetv2/__init__py 内容
from .scannetv2_inst import ScanNetV2Inst
__all__ = [k for k in globals().keys() if not k.startswith("_")]
```
以上可以看到 `__init__.py` 起到的作用相当于将 `ScanNetV2Inst` 的调用一层一层地向外搬。最终实现了：
```python
import gorill3d
dataset_caller = gorilla3d.ScanNetV2Inst
```
所以当同学们在相应的文件写好了相应的代码后应该对以上的内容进行补充，例如我需要补充一个 `ShapeNetCls` 的 ShapeNet 分类数据集，则因对 `dataset` 进行补充：
```sh
datasets/
├── __init__.py
├── scannetv2
│   ├── __init__.py
│   └── scannetv2_inst.py
├── shapenet # 添加以数据集为名称的目录
│   ├── __init__.py # 需要编写的 __init__.py 文件
│   └── shapenet_cls.py # 加上任务后缀并在内部定义数据集接口
└── utils
    ├── __init__.py
    ├── pc_aug.py
    └── transforms.py
```
那么在 `datasets/shapenet/shapenet_cls.py` 中定义好 `ShapeNetCls` 数据接口类后，应编写 `datasets/shapenet/__init__.py`：
```python
from .shapenet_cls import ShapeNetCls
__all__ = [k for k in globals().keys() if not k.startswith("_")]
# __all__ 不要求大家了解，有兴趣的可以查找相关资料自行了解
```
然后补充 `datasets/__init__.py`：
```python
from .scannetv2 import ScanNetV2Inst
from .shapenet import ShapeNetCls # 需要补充的内容
__all__ = [k for k in globals().keys() if not k.startswith("_")]
```
以上就完成了数据集的添加，同时完善了相关的 `__init__.py` 实现了其简单调用。

## python setup.py develop
相信很多同学在安装包的时候最常用的操作是 `pip install package` 亦或者是下载源码后运行 `python setup.py install` 安装。
这里要介绍的则是 `python setup.py develop` 这个命令，估计很多同学没有了解过这个命令，从后缀可以看得出来这里这样的安装模式是为开发者而设计的，而开发者需要的一个很重要的要求就是修改了某个内容后能够及时地得到反馈，以上面的例子来说就是当我往 `gorilla3d` 中添加了 `ShapeNetCls` 后，我要能马上验证：
```python
import gorilla3d
dataset_caller = gorilla3d.ShapeNetCls
```
这里要说一下 Python 调用 package 时的原理，当运行 `import gorilla3d` 时会去搜索 `site-packages` 中名为 `gorilla3d` 的目录，如果是 `pip install ` 和 `python setup.py install` 则是进行相应的编译安装后将目录复制到 `site-packages` 中，如果是 `python setup.py develop` 则会在源码目录进行相应的编译安装，然后创造一个软连接到 `site-package` 中，这样在源码的任何操作都能直接反应到当前使用的环境中，无需跑到 `site-packages` 中修改。



