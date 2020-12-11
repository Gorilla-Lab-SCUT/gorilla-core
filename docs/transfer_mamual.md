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
            print("set random seed:", seed)
            set_random_seed(seed, logger=self.logger)

    def resume(self, checkpoint, **kwargs):
        # 加载权重的API
        check_file_exist(checkpoint) # 检查文件是否存在
        # 加载权重，同时读取 meta 信息
        self.meta = resume(self.model,
                           checkpoint,
                           self.optimizer,
                           self.lr_scheduler)
        # 如果 meta 中有epoch信息，则写入当前solver
        if "epoch" in self.meta:
            self.epoch = self.meta["epoch"]

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


