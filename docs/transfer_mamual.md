<!-- Copyright (c) Gorilla-Lab. All rights reserved. -->

# 代码迁移手册
该手册的目的在于辅助同学们按照相应的模式进行代码的迁移。

---
## solver
首先是 `solver` 部分，该部分在于实现训练Pipeline的搭建，在以后的项目中希望同学们将网络训练部分的代码以 `solver`**类** 的形式进行包装，而不是分散地写在脚本中。
在我们的核心库`gorilla-core`中提供了`solver`的范式`BaseSolver`：
```python
class BaseSolver(metaclass=ABCMeta):
    def __init__(self,
                 model,
                 dataloaders,
                 cfg,
                 logger=None,
                 **kwargs):
        # 初始化必要成员
        self.model = model
        self.dataloaders = dataloaders
        # meta 用于存放有需要的参数，例如时间/epoch数/正确率等,
        self.meta = {}
        # 构建优化器以及学习率策略（具体可以看源码实现）
        self.optimizer = build_optimizer(model, cfg.optimizer)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, cfg.lr_scheduler)
        # 读取配置文件
        self.cfg = cfg
        # 初始化logger
        self.logger = logger
        # 初始化起始epoch和iter
        self.epoch = cfg.get("start_epoch", 1)
        self.iter = 0
        # 初始化记录容器，用于写入tensorboard，要了解原理观看源码
        self.log_buffer = LogBuffer()
        # tensorboard 容器
        self.tb_writer = SummaryWriter(log_dir=cfg.log) 

        # 准备训练前的设置
        self.get_ready()

    def get_ready(self, **kwargs):
        # NOTE: 该函数的目的在于进行训练前设置（可自行定义）
        # 根据配置文件设置随机数种子
        seed = self.cfg.get("seed", 0)
        if seed != 0:
            from ..core import set_random_seed
            set_random_seed(seed, logger=self.logger)

    def resume(self, checkpoint, **kwargs):
        # 加载权重的API
        check_file_exist(checkpoint) # 检查文件是否存在
        # 加载权重，同时读取 meta 信息
        self.meta = resume(self.model,
                           checkpoint,
                           self.optimizer,
                           self.lr_scheduler,
                           **kwargs)
        # 如果 meta 中有epoch信息，则写入当前solver
        if "epoch" in self.meta:
            self.epoch = self.meta["epoch"] + 1

    def write(self, **kwargs):
        # NOTE: 由于大家使用tensorboard的习惯不同，这里仅实现了最
        #       基本的从 dict 利用 add_scalar 写入 tensorboard
        #       如果有别的需求复写这个函数即可
        self.logger.info("Epoch: {}".format(self.epoch))
        # 对记录容器中的值求取均值然后写入tensorboard中
        self.log_buffer.average()
        for key, avg in self.log_buffer.output.items():
            self.tb_writer.add_scalar(key, avg, self.epoch)

    def clear(self, **kwargs):
        # 清空记录容器（例如每个epoch清空以便记录下一个epoch的数据）
        self.log_buffer.clear()

    @abstractmethod
    def solve(self, **kwargs):
        self.clear()
        # solver启动函数，对应整个训练流程

    @abstractmethod
    def train(self, **kwargs):
        self.clear()
        # 进行每个epoch的训练，根据自己任务编写

    @abstractmethod
    def evaluate(self, **kwargs):
        # 验证网络，根据自己任务编写
        self.clear()
    
```
我们提供了`BaseSolver`这个范式，其中`train/evaluate`分别对应同学们训练脚本中的训练和验证部分也是同学们要自行编写的主体。
关于编写规范，这里介绍以一个非常简单的实现为例：
```python
import gorilla
class SpecificSolver(gorilla.BaseSolver):
    ...
    # solve 函数其功能可以看作是训练的启动器和管理器，不负责具体实现
    def solve(self):
        while self.epoch < self.cfg.max_epoch:
            # epoch训练
            self.train()
            if self.val_flag: # 是否触发验证条件
                self.evaluate()
            self.epoch += 1

    # train 函数可以看作 low-level的solve，不过其重点在于遍历
    # 数据集以及对网络前传的结果进行处理
    def train(self):
        # 清空记录容器
        self.clear()
        for i, (data, gt) in enumerate(self.train_data_loader):
            # 具体网络前传以这种形式实现
            loss = self.step(data, gt)
            # 反传及更新参数
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
        
        # NOTE: 自定义保存的位置并保存权重
        checkpoint = get_save_path(...)
        gorilla.save_checkpoint(self.model,
                                checkpoint,
                                self.optimizer,
                                self.lr_scheduler,
                                self.meta)
        # 写入tensorboard
        self.write()

    # step 函数专注于网络的前传以及参数的获取记录
    def step(self, data, gt, mode="train"):
        # 每一个iter的前传操作
        # 解析获取数据
        data = data.cuda()
        gt = gt.cuda()
        # 网络前传
        predict = self.model(data)
        # 计算loss
        loss, acc = self.criterion(predict, gt)
        # 获取记录的参数并传递给记录容器
        param_dict = {"loss": loss,
                      "acc": acc}
        self.log_buffer.update(param_dict)
    
    def evaluate(self):
        self.clear()
        # 根据自己的需求写验证函数
        ...
```
训练脚本方便希望同学们根据自己的情况套以上模板，由于同学们测试条件和处理各异，测试脚本则不对同学们做要求。

## config & argparse
如何管理实验流程，大体上无非就两种方式，一种是配置文件，将其读取成 `dict` 对象后从中索引参数；第二种就是 `Argparse`，通过命令行输入以及默认的设置来确定参数。
大部分情况则是两者进行结合，配置文件确定不常改变的参数例如optimizer参数，数据集路径等，`Argparse`则是用来设置较为灵活的参数，例如学习率，保存地址等。
这里的话我们也是提供了辅助同学们进行配置管理的工具函数。
在配置管理方面，我们这里对同学们提四个要求：
1. 配置文件要有分层，而且至少有以下分层（以 `.yaml` 为例）：
```yaml
model:
    ...
data:
    ...
optimizer:
    ...
lr_scheduler:
    ...
```
可以有多的子配置项，但是这四个子配置项是不可少的。

1. 其中的子配置项 `optimizer` 以及 `scheduler` 要符合以下要求（ `model` 以及 `data` 由于同学们需求各异，则不对内部参数进行统一要求）：
```yaml
optimizer:
    name: Adam,
    lr: 0.001
lr_scheduler:
    name: StepLR,
    step_size: 10000
```
这里希望同学们统一使用 `torch.optim.lr_scheduler` 中的 `_LRScheduler` 派生的各种学习率策略实现，尽量不要使用自己的学习率函数，如果有自己的需求可以基于 `_LRScheduler` 派生出自己的学习率策略并贡献到代码库（例如我们就利用这种方式实现了 `PolyLR` 以及带有各种预热机制的学习率策略）或者使用 `LambdaLR` 实现自定义学习率，以上的解决方案基本能够覆盖学习率策略的不同情况。
上面 `optimizer/lr_scheduler` 的配置格式是为了配合我们的 `build_optimizer/build_lr_scheduler` 函数使用，在上面提到的`BaseSolver`中这样调用：
```python
class (metaclass=ABCMeta):
    def __init__(self, **kwargs):
        ...
        # 构建优化器以及学习率策略（具体可以看源码实现）
        self.optimizer = build_optimizer(model, cfg.optimizer)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, cfg.lr_scheduler)
        ...
```
这两个构建函数的原理也非常简单，根据 `name` 成员获取相应的声明后剩余的参数结合 `model/optimizer` 作为初始化的参数得到相应的 `optimizer/lr_scheduler`。
以 `build_optimizer(model, cfg.optimizer)` 为例，根据 `name: "Adam"` 获得了 `torch.optim.Adam` 的声明，其 API 如下：
```python
torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```
对于大部分同学来说要负责的参数是 `params` 和 `lr`，所以 `build_optimizer` 内部会获取 `model` 中的参数作为 `params`，紧接着把配置文件中的 `cfg: 0.001` 作为 `lr`，如果要给的那个 `betas` 假设为 `[0.9, 0.99]`，那么配置文件改为以下即可：
```yaml
optimizer:
    name: Adam,
    lr: 0.001
    betas: [0.9, 0.99]
```
`build_lr_scheduler` 同理，它根据 `name` 获取相应的声明，接受 `optimizer` 为首位参数，剩余的从配置文件中读取。

3. 以上配置文件尽可能使用我们的 `Config` 类进行管理，这部分我们直接使用了 `mmcv` 提供了 `Config` 类，所以是有 bug free 的保证的，我们也会，里面提供了非常好的性质其中主要函数为 `Config.fromfile` 同学们可通过阅读之前的介绍手册了解。
4. 同学们在获取 `cfg` 和 `args` 后应尽早将两者合成一个 `cfg`，保证后面的 pipeline 只通过一个 `cfg` 管理配置参数。`args` 作为 `argparse.NameSpace` 可以通过Python内置的 `vars(.)` 转换成 `dict`。
   当然为了辅助同学们进行融合，我们也提供了 `merge_cfg_and_args` 函数，直接将 `cfg` 和 `args` 合并成一个总的 `cfg`，本质就是将 `args` 中的参数融合进 `cfg` 中，如果有重合的参数，则以 `args` 中的参数对 `cfg` 中的参数进行覆盖。


