# gorilla-core 常用功能函数及介绍

gorilla 是一个结合了 mmcv 和 detectron2 的基础库，目前主要是和 deep learning framework 无关的一些工具函数，以及一些辅助训练工具。
- 下面介绍一下常用的函数。
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
gorilla.dump(data, "out.pkl")

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
- **时间统计**

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

- **过程统计**
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

- **配置管理**

