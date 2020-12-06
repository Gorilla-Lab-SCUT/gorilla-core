# gorilla-core 常用功能函数及介绍

gorilla 是一个结合了 mmcv 和 detectron2 的基础库，目前主要是和 deep learning framework 无关的一些工具函数，以及一些辅助训练工具。该基础库的文件目录如下：

```sh
gorilla
    ├── core
    ├── utils
    ├── fileio
    ├── config
    ├── evaluation
    ├── examples
    ├── losses
    ├── nn
    ├── solver
    ├── __init__.py
    └── version.py
```


 下面介绍一下常用的函数。
---
## fileio
`fileio` 模块支持直接对 `.json`, `.yaml`, `.pkl` 的加载和读取
```python
import gorilla

####### 支持直接对 `.json`, `.yaml`, `.pkl` 的加载和读取 #######

#直接从文件加载
#可以加载json,yaml,pkl文件
data = gorilla.load("test.json")
data = gorilla.load("test.yaml")
data = gorilla.load("test.pkl")
    
# 将数据转储为文件
gorilla.    (data, "out.pkl")

# 从一个文件类别加载
with open("test.json", "r") as f:
    data = gorilla.load(f)

# 将文件转储为字符串
json_str = gorilla.dump(data, file_format="json")

# 从一个文件类别储存
with open("test.yaml", "w") as f:
    data = gorilla.dump(data, f, file_format="yaml")
```
该模块还支持加载文本文件为作为 `list` 或 `dict`（文本内容需要符合相应格式）
- `list_from_file`
  
假设存在文本 `a.txt`:
```txt
a
b
c
d
e
```
通过 `list_from_file` 可以实现：
```python
>>> gorilla.list_from_file("a.txt")
["a", "b", "c", "d", "e"]
>>> gorilla.list_from_file("a.txt", offset=2)
["c", "d", "e"]
>>> gorilla.list_from_file("a.txt", max_num=2)
["a", "b"]
>>> gorilla.list_from_file("a.txt", prefix="/mnt/")
["/mnt/a", "/mnt/b", "/mnt/c", "/mnt/d", "/mnt/e"]
```

- `dict_from_file`

假设存在文本 `b.txt`:
```txt
1 cat
2 dog cow
3 panda
```
通过 `dict_from_file` 可以实现：
```python
>>> gorilla.dict_from_file("b.txt")
{"1": "cat", "2": ["dog", "cow"], "3": "panda"}
>>> gorilla.dict_from_file("b.txt", key_type=int)
{1: "cat", 2: ["dog", "cow"], 3: "panda"}
```

## utils
`utils` 模块提供了许多辅助性的工具函数。
### **时间统计**

使用 `Timer` 可以非常方便地对运行时间进行截取：
```python
>>> import time
>>> import gorilla
>>> with gorilla.Timer():
>>>     time.sleep(1) # 经过 1s
1.000
>>> with gorilla.Timer(print_tmpl="it takes {:.1f} seconds"):
>>>     time.sleep(1) # 经过 1s
it takes 1.0 seconds
>>> timer = gorilla.Timer()
>>> time.sleep(0.5)
>>> print(timer.since_start()) # 在这里截取一个时间节点
0.500
>>> time.sleep(1.0)
>>> print(timer.since_last()) # 计算该节点与上一个节点的时间差
1.000
>>> print(timer.since_start()) # 计算和开始节点的时间差
1.500
```

同时也提供了一个时间戳函数 `check_time` 以方便在循环中获取运行时间：
```python
>>> import time
>>> import gorilla
>>> for i in range(1, 5):
>>>     time.sleep(i)
>>>     print(gorilla.check_time("task1"))
0.000
2.000
3.000
4.000
```
以上生成的时间戳识别器会可以通过 `gorilla.utils.timer._g_timers["task1"]` 获取。

### **过程统计**

（对 `tqdm` 模块更熟悉的同学可以使用 `tqdm` 模块）
该模块提供 `ProgressBar` 来对过程进度进行跟踪：
```python
>>> import time
>>> import gorilla
>>> # example function
>>> def plus_one(n):
>>>     time.sleep(0.5)
>>>     return n + 1
>>>
>>> prog_bar = gorilla.ProgressBar()
>>> for i in range(10):
>>>     plus_one(i)
>>>     prog_bar.update() # 手动更新
[>>>>>>>>>>>>>>>          ] 6/10, 2.0 task/s, elapsed: 2s, ETA: 2s
```

如果要将方法应用于项目列表并跟踪进度，那么 `track_progress` 是一个不错的选择。它对 `ProgressBar` 进行了稍微的包装：
```python
>>> tasks = range(10)
>>> gorilla.track_progress(plus_one, tasks)
[>>>>>>>>>>>>>>>          ] 6/10, 2.0 task/s, elapsed: 2s, ETA: 2s
```

还有另一个方法 `track_parallel_progress`，它还包装了并行处理（需要执行函数支持并行）。
```python
>>> gorilla.track_parallel_progress(func, tasks, 8)  # 8 workers
```

还有顺带初始化多线程池的函数 `init_pool`，可以通过源代码查看细节。

以及与 `tqdm.tqdm` 具有相似功能的 `gorilla.track`：

```python
>>> import gorilla
>>> for i in gorilla.track(range(10)):
>>>     pass
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 10/10, 35484.8 task/s, elapsed: 0s, ETA:     0s
>>> import tqdm
>>> for i in tqdm.tqdm(range(10)):
>>>     pass
100%|██████████████████████████████| 10/10 [00:00<00:00, 74499.18it/s]
```

### **GPU管理**
  
该模块提供了**gpu监视和自动功能索引函数**。
`get_free_gpu` 函数可以获取当前满足条件的 **gpu的id索引列表**，默认为检索空余显存超过11G的gpu，如果需要检索空闲（无占用程序）的gpu，则设置 `mode="process"` 即可。
```python
def get_free_gpu(mode="memory", memory_need=11000) -> list
```

在此基础上，如果要监视多gpu，我们则提供了 `supervise_gpu` 函数，`num_gpu` 为需要获取的gpu的数量，`mode` 与 `memory_need` 同上，当该程序发现有该数量符合条件的gpu时，则返回这些gpu的id索引列表，否则一直等待，直到有空闲gpu满足 `num_gpu` 的数量。
```python
def supervise_gpu(num_gpu=1, mode="memory", memory_need=11000) -> list
```

最后还有自动设置 `CUDA_VISIBLE_DEVICES` 的函数，通过设置 `os.environ["CUDA_VISIBLE_DEVICES"]` 实现：
```python
def set_cuda_visible_devices(gpu_ids=None, num_gpu=1, mode="memory", memory_need=11000)
```
当 `gpu_ids` 给定，则设置为给定的 `gpu_ids` 为 `os.environ["CUDA_VISIBLE_DEVICES"]`，否则调用 `supervise_gpu` 函数获取符合空闲条件的gpu，直接设置。可以免除 `CUDA_VISIBLE_DEVICES=x python script.py` 的前缀工作。

### **路径管理**

`gorilla.utils.path` 中定义了些许函数，本质上是对 `os` 中一些函数的包装，其中有可以递归遍历文件夹获取相应后缀的文件列表的 `scandir` 函数：
```python
def scandir(dir_path, suffix=None, recursive=False)
```
指定遍历根目录 `dir_path`，就可以搜索符合后缀为 `suffix`（可以为包含多个后缀的 `tuple`） 的文件，`recursive=True` 则递归搜索完所有的子文件夹，最终返回为 `generator`，可以通过 `list(.)` 转为列表。

### **日志管理**

python的logging库已经非常完善和易用了，这里仅介绍函数 `get_root_logger`，实现功能也非常简单，本质上调用了`logging.getLogger` 然后对里面一些参数和格式进行了设置。
```python
def get_root_logger(log_file=None, log_level=logging.INFO, timestamp=None)
```

### **模型读写管理**

该模块还提供了模型读写管理的功能函数。
对于使用 `DataParallel/DistributedDataParallel` 包装并行的网络的名称前缀都会有 `.module` ，对于这种情况，无论时保存和读写都需要在加载或者保存时进行前缀的处理，我们提供的函数则帮你处理了这些繁琐的工作。

有 `is_module_wrapper` 来判断是否对其进行了并行的包装，进而在保存时，
仅保存其 `.module` 部分，也就是把 `.module` 去掉了；在加载时，则仅将网络加载进 `.module` 部分。

- 保存

保存网络的函数为 `save_checkpoint`，将模型参数保存为 `filename`：
```python
def save_checkpoint(model, filename, optimizer=None, scheduler=None, meta=None)
```
该函数还支持保存对 `optimizer` 以及 `lr_scheduler` 进行保存，以便在下次导入训练时还原训练的关键参数，在这里保存的 `dict` 键值索引名称如下：
```python
checkpoint = {
    "state_dict": 网络参数,
    "optimizer": optimizer参数,
    "scheduler": lr_scheduler参数,
    "meta": 存放任意参数的字典，例如时间/epoch数等,
}
```
当保存对象输入为 `None` 默认保存为 **空字典**。

- 读取

读取网络的部分这里介绍两个函数： `load_checkpoint` 和 `resume`。
`load_checkpoint` 是仅针对网络参数加载的函数。
```python
def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None)
```
支持从 `url` 下载所需的权重。在支持直接导入模型参数的基础上（既`checkpoint`本身就是`state_dict`），也为了支持上述的 `checkpoint` 键值索引也进行了相应的判断和处理。

另外一个函数就是对 `load_checkpoitn` 函数的扩展和包装 `resume`。当我们训练网络中断，继续训练的时候，我们在已经保存 `optimizer` 和 `lr_scheduler` 的基础上，需要把它们也加载进来。`resume` 函数就可以看作在 `load_checkpoint` 的基础上实现 `load_optimizer` 和 `load_lr_scheduler` 的功能：
```python
def resume(model,
           filename,
           optimizer=None,
           scheduler=None,
           resume_optimizer=True,
           resume_scheduler=True,
           map_location="default")
```

## **显存试错**
这里仅涉及到一个函数，来自 `detectron2` 库，`retry_if_cuda_oom`，这个函数可以可以看作对函数的包装函数，其功能在于在一定程度上避免OOM的情况
```python
def retry_if_cuda_oom(func)
```
这里仅放一个 `detectron2` 中的使用案例，这个是2d检测任务，对生成的 `anchor` 与 `gt_bboxes` 进行匹配，由于 `anchor` 的数量很多所以可能会出现OOM的情况，当捕获到OOM异常时，会先执行`torch.cuda.empty_cache()`操作再进行尝试，如果依旧OOM，则将其放到cpu上运行。
不过如果是网络过大输出过大特征图导致的OOM则无效，仅对索引函数有效。
```python
match_quality_matrix=retry_if_cuda_oom(pairwise_iou)(gt_boxes_i,anchors_i)
```

# Config
该模块提供了非常实用的配置类`Config`。
它支持从多种文件格式（包括 `.py`，`.json`  `.yml` 和 `.yaml`）加载配置。加载进来的配置类`Config`与`dict`有相似的性质，更方便的是它不仅可以用`config["key"]` 的方式索引，更可以通过 `config.key` 的方式索引，也支持 `**config` 实现函数参数的键值传递。
```python
a.yaml
#########################
a: 1
b: {"b1": [0, 1, 2], "b2": None}
c: (1, 2)
d: "string"
##########################

#测试案例
cfg = Config.fromfile("a.yaml")
assert cfg.a == 1
assert cfg.b.b1 == [0, 1, 2]
cfg.c = None
assert cfg.c == None
```

另外就是该类支持非常好的融合性质，我们的网络实际上由非常多的部分组成，实际上我们的配置文件往往包含了网络的所有参数，有时候不太方便。
`Config` 支持在配置文件中定义 `_base_` 对象，`_base_` 对象中存放的是需要融合并覆盖的子配置文件。
```python
b.json(".json"文件，但对".py", ".json", ".yaml"都支持)
#########################
{
    "_base_": "./a.yaml", # 融合对象是 ".yaml" 文件，定义在上
    "c": [3, 4],
    "d": "Str"
}
##########################

#测试案例
>>> cfg = Config.fromfile("b.json")
>>> print(cfg)
Config (path: b.json): {
    "a": 1,
    "b": {"b1": [0, 1, 2], "b2": "None"},
    "c": [3, 4],
    "d": "Str"
    }
```

可以看到加载 `b.json` 对象后得到的 `Config` 在继承了 `a.yaml` 中的成员后对已有的 `c`，`d` 成员进行了覆盖。

同时，该 `Config` 在初始化`dict`（加载文件暂时未实现）会自动地根据 `.` 进行层级划分
```python
>>> options = {"model.backbone.name": "ResNet",
>>>            "model.backbone.depth": 50}
>>> cfg = gorilla.Config({"model": {"backbone": {"name": "VGG"}}})
>>> cfg.merge_from_dict(options)
>>> print(cfg)
Config (path: None): {
    "model":{
        "backbone": {
            "name": "ResNet",
            "depth": 50}}}
```
但是在加载的时候`Config(dict)`和`Config.fromfile(filename)`并不会自动进行层级划分，需要注意。另外就是上面例子中提到的 `merge_from_dict` 成员函数，它可以根据融合对象对已有的配置进行融合覆盖，上面的例子就表明了，`name` 这个成员原本为 `VGG` 被 `ResNet` 覆盖了。

另外就是许多同学非常喜欢使用 `argparse` 管理超参数，为了方面管理我们希望实现 `cfg` 和 `args` 的统一，经过我们的思考，我们提供了 `merge_args_and_cfg` 函数，实现融合：
```python
def merge_args_and_cfg(cfg: Optional[Config]=None, args: Optional[ArgumentParser]=None) -> Config
```
输入分别为 `cfg` 和 `args` 融合得到新的 `cfg`，由于 `args` 中的参数优先度往往比 `cfg` 中的参数高，所以我们利用了上面所说的 `merge_from_dict` 函数实现了两者的融合，对于相同的参数，则利用 `args` 中的参数进行覆盖。

# Core
Core 作为代码库的核心，里面包含了许多必要的函数，其中也包括很多杂项函数，这一部分我们还在整理中，目前对外有两个设置的函数。
一个是设置随机数种子的 `set_random_seed`
```python
set_random_seed(seed, deterministic=False, use_rank_shift=False)
```
通常来说只用给定 `seed` 即可，里面本质操作就是分别设置 `np/torch/random` 的 `seed`，在这里只用通过一行代码即可解决。

另一个函数则是用于收集环境信息的函数 `collect_env_info`，该函数不用任何输入，运行后直接返回当前环境信息的字符串：
```python
>>> import gorilla
>>> print(gorilla.collect_env_info())
-------------------  ------------------------------------------------------------------------------------------
sys.platform         linux
Python               3.7.0 (default, Oct  9 2018, 10:31:47) [GCC 7.3.0]
numpy                1.19.2
gorilla              0.2.3.6 @/data/lab-liang.zhihao/code/gorilla-core/gorilla
GORILLA_ENV_MODULE   <not set>
PyTorch              1.3.0 @/home/lab-liang.zhihao/miniconda3/envs/pointgroup/lib/python3.7/site-packages/torch
PyTorch debug build  False
GPU available        True
GPU 0,1,2,3,4,5,6,7  GeForce RTX 2080 Ti (arch=7.5)
CUDA_HOME            /usr/local/cuda-10.0
torchvision          unknown
cv2                  4.4.0
-------------------  ------------------------------------------------------------------------------------------
PyTorch built with:
  - GCC 7.3
  - Intel(R) Math Kernel Library Version 2020.0.2 Product Build 20200624 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v0.20.5 (Git Hash 0125f28c61c1f822fd48570b4c1066f96fcb9b2e)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - NNPACK is enabled
  - CUDA Runtime 10.1
  - NVCC architecture flags: -gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_50,code=compute_50
  - CuDNN 7.6.3
  - Magma 2.5.1
  - Build settings: BLAS=MKL, BUILD_NAMEDTENSOR=OFF, BUILD_TYPE=Release, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -fopenmp -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -O2 -fPIC -Wno-narrowing -Wall -Wextra -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Wno-stringop-overflow, DISABLE_NUMA=1, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, USE_CUDA=True, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, USE_STATIC_DISPATCH=OFF,
```
剩下的函数我们也在进行整理。

# solver
该模块主要是设计网络训练的辅助函数。
- **学习率策略**

在训练的时候我们希望大家的学习率调整策略尽量使用 `torch.optim.lr_scheduler` 中提供的 `scheduler` 实现，如果是自己写的学习率变化函数也尽量使用 `torch.optim.lr_scheduler.LambdaLR` 进行包装。我们在已有的学习率策略的基础上还提供了四种学习率策略分别是：
```python
WarmupMultiStepLR, WarmupCosineLR, WarmupPolyLR, PolyLR, InvLR
```
它们都是继承自 `torch.optim.lr_scheduler._LRScheduler`，如果有同学有新的学习率策略是原本没有的，希望可以遵循相应的格式贡献到代码库中。

- **优化器和学习率策略构建函数**

另外，我们也提供了非常轻量级的构建函数，分别是：
```python
def build_single_optimizer(
        model: torch.nn.Module,
        optimizer_cfg: [Config, Dict]) -> torch.optim.Optimizer

def build_lr_scheduler(
        optimizer: torch.optim.Optimizer,
        lr_scheduler_cfg: [Config, Dict],
        lambda_func=None) -> torch.optim.lr_scheduler._LRScheduler
```
其中的 `optimizer_cfg` 和 `lr_scheduler_cfg` 分别是传给 `Optimizer` 和 `xxxLR` 的键值对，至于要调用哪个 `Optimizer` 和 `xxxLR`，则在 `cfg` 里面定义好 `name` 即可

```python
>>> import gorilla
>>> model = gorilla.VGG(16)
>>> # 构建optimizer
>>> optimizer_cfg = {"name": "Adam", "lr": 0.002}
>>> optimizer = gorilla.build_optimizer(model, optimizer_cfg)
>>> optimizer
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.002
    weight_decay: 0
)
>>> # 构建lr_scheduler
>>> scheduler_cfg = {"name": "MultiStepLR", "milestones": [30, 80], "gamma": 0.1}
>>> scheduler = gorilla.build_lr_scheduler(optimizer, scheduler_cfg)
>>> scheduler
<torch.optim.lr_scheduler.MultiStepLR at 0x7f7da41f99e8>
```
当这些写入配置文件中就可以非常方便的进行构建了。
> NOTE: `build_optimizer` 可以用 `build_optimizer_v2` 代替，原本功能保持不变的同时支持多 `Optimizer` 的构建。

- **梯度裁剪器**

另外针对梯度裁剪的需求，我们也提供了 `GradClipper` 类似上面构建梯度裁剪器，以及 `build_grad_clipper` 的构建接口。
```python
>>> import gorilla
>>> grad_clip_cfg = {"name": "norm", "max_norm": 20}
>>> clipper = gorilla.GradClipper(**grad_clip_cfg)
>>> clipper = gorilla.build_grad_clipper(**grad_clip_cfg)
>>> ...
>>> loss.backward()
>>> grad_norm = clipper.clip(model.parameters())
>>> optimizer.step()
```
`clip` 成员函数本质上是调用 `torch.nn.utils.clip_grad.clip_grad_{norm/value}_` 函数，熟悉的同学也可以直接调用这个函数。

- **pipeline 管理**

另外针对训练的 `pipelipe`，我们也提供了一个非常基础的基类 `BaseSolver`，里面提供了一些非常简单的接口，希望同学们的 pipeline 可以继承该 `Solver` 进行复写，由于每个人任务不同，需要的功能很可能区别很大，因此我们不强行规定 `pipeline`，希望以后同学们能够形成统一的规范，我们也能对这部分代码进行更好地整合。

- **日志变量存储器**

如何保存日志有时候需要写些循环和赋值运算操作，在这里我们提供了两个类方便变量的存储。
一个是针对单个变量的 `HistoryBuffer`，无需参数直接初始化即可。当需要更新的时候调用 `update` 函数：
```python
def update(self, value: float, num: Optional[float] = None) -> None
```
输入值以及该值的数量（相当于比重，默认为`1`），然后输入的值和数量分别存在 `list` 中，后续在算 `avg` 时会根据数量进行加权得到。
```python
>>> import gorilla
>>> buffer = gorilla.HistoryBuffer()
>>> buffer.update(10)
>>> buffer.update(12)
>>> buffer.update(14, 2)
>>> buffer.update(15)
>>> buffer.avg
13.0 # (10 + 12 + 14 + 15) / (1 + 1 + 2 + 1)
>>> buffer.values
[10, 12, 14, 15]
>>> buffer.nums
[1, 1, 2, 1]
>>> buffer.latest
15
>>> buffer.average(3)
13.75 # (12 + 14 + 15) / (1 + 2 + 1) # 求values后三个的均值
>>> buffer.average(3) # 求values后三个的中位数
14.0
```
在此基础上我们提供了 `LogBuffer` 类结合 `HistoryBuffer` 实现多个变量的列表管理。`LogBuffer` 可以看作是值成员为 `HistoryBuffer` 的字典，`LogBuffer` 的 `update` 为字典。
```python
>>> import gorilla
>>> buffer = gorilla.LogBuffer()
>>> buffer.update({"a": 10, "b": [10, 2]})
>>> buffer.update({"a": 12, "b": [12, 3]})
>>> buffer.average() # 调用HistoryBuffer的avgerate计算均值
>>> buffer.output
OrderedDict([('a', 11.0), ('b', 11.2)])
>>> buffer.get("b")
<gorilla.solver.log_buffer.HistoryBuffer at 0x7f8cbf4f59b0>
>>> buffer.get("b").values
[10.0, 12.0]
>>> buffer.get("b").nums
[2, 3]
>>> buffer.clear()
>>> buffer.get("b")
None
```





