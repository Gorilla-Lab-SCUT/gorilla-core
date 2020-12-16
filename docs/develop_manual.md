# gorilla2d/3d 开发指南

我们建立 `gorilla2d/3d` 的目的在于更好地对实验室的代码进行管理，同时，我们的代码库处于比较初始的阶段，为了更好地在辅助同学们开发的同时建立我们的代码库，我们希望同学们在平时开发的过程中尽可能地利用我们代码库已有的东西，同时也尽可能地向代码库贡献代码。
如何利用 `gorilla-core` 辅助开发在 `transfer_manual` 中已经进行了说明，接下来该手册的目的在于如何更好地进行开发。

---


## 注意事项

- **切记！commit内容一定要写好，commit内容必须能够反应当前commit进行了什么操作或者实现了什么功能**，鼓励同学们少量多次的提交，而不是一股脑的写好后一次性提交，也不方便我们进行review。
- **一定要在`dev`分支上进行更新，`master` 分支不进行开发！**
- `Dataset` 的 `__getitem__` 函数返回的统一包装成 `dict` 形式，网络的输出也统一包装成 `dict` 形式。
- **最后，一定要写注释！！！简单的注释即可，可以参考 `transfer_manual`**


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


## 开发要求
下面就来说明同学们在开发过程中需要注意的一些事情。需要同学们参与开发的库分别是 [gorilla2d](http://222.201.134.203:20818/GorillaLab/gorilla-2d) 和 [gorilla3d](http://222.201.134.203:20818/GorillaLab/gorilla-3d)，同学们根据自己的任务进行选择。
- 首先需要同学们从主仓库中 `fork/派生` 得到自己的子仓库，之后贡献代码的方式需要同学们更新自己的仓库后 `pull&request/创建合并请求` 到主仓库从而更新代码库，不能直接对主仓库，代码进行更新。
- `git clone` 源码后运行 `python setup.py develop` 进行编译安装，`gorilla3d` 中存在些许 `CUDA/C++` 拓展，所以安装会比较慢，然后就可以在本地修改和贡献相应的代码了。`Git` 的相关操作这里暂不进行说明，需要同学们自行了解。
- 同学们应该操作的分支为 `dev` 分支，`master` 分支一般不进行直接操作，而是等 `dev` 分支处于某个稳定节点后进行融合更新，同学们 `fork/派生` 到自己的子仓库后也应该操作 `dev` 分支，`pull&request/创建合并请求` 操作也应该是请求 `dev` 分支的合并。
- 同学们安装好上述代码库后，可以新建目录作为新项目，也可以在相应的 `gorilla2d(3d)/projects` 下新建目录作为新项目，这里不做要求，反正进行上述的操作后就可以实现 `import gorilla2d(3d)`。
- 什么东西应该放代码库里，其实并不要求同学们所有的代码都融入到代码库中，也不要求同学们所有的代码能替换的都使用我们的代码库进行替换，需要融入到代码库的东西有 `dataset/models/evaluator`。`dataset` 方面无需多说，一般研究进行到一定程度，数据集会是一个比较稳定的版本，不需要进行大幅的更改或者更改仅需参数即可实现，这部分的话就可以贡献到代码库中。另外 `evaluator` 的设计参考 `gorilla2d/3d` 的文档保证复用性。

## 注意事项（最后再重复一次）

- **切记！commit内容一定要写好，commit内容必须能够反应当前commit进行了什么操作或者实现了什么功能**，鼓励同学们少量多次的提交，而不是一股脑的写好后一次性提交，也不方便我们进行review。
- **一定要在`dev`分支上进行更新，`master` 分支不进行开发！**
- `Dataset` 的 `__getitem__` 函数返回的统一包装成 `dict` 形式，网络的输出也统一包装成 `dict` 形式。
- **最后，一定要写注释！！！简单的注释即可，可以参考 `transfer_manual`**
