## 一、Qlib安装

1.安装Qlib

直接通过pip安装Qlib

```powershell
pip install pyqlib
```

2.安装torch

通过pip直接安装

```powershell
pip install pytorch
```



## 二、运行模型

### 1.下载数据集

​	根据官网提示，使用qlib的getData指令下载数据集至本地

```powershell
python scripts/get_data.py qlib_data --target_dir S:\Coding\Worspace\Transformer\Qlib\dataset --region cn
```

​	下载完成后，按照文档提示检视数据，数据可以正常显示。

```python
>>> import qlib
>>> from qlib.data import D
>>> qlib.init(provider_uri="S:/Coding/Worspace/Transformer/Qlib/dataset")
[16908:MainThread](2022-06-27 17:20:15,887) INFO - qlib.Initialization - [config.py:413] - default_conf: client.
[16908:MainThread](2022-06-27 17:20:16,270) INFO - qlib.Initialization - [__init__.py:74] - qlib successfully initialized based on client settings.
[16908:MainThread](2022-06-27 17:20:16,270) INFO - qlib.Initialization - [__init__.py:76] - data_path={'__DEFAULT_FREQ': WindowsPath('S:/Coding/Worspace/Transformer/Qlib/dataset')}
>>> instruments = D.instruments(market='csi300')
>>> print(D.list_instruments(instruments=instruments, start_time='2010-01-01', end_time='2017-12-31', as_list=True)[:20])
['SH600000', 'SH600004', 'SH600009', 'SH600010', 'SH600011', 'SH600015', 'SH600016', 'SH600018', 'SH600019', 'SH600027', 'SH600028', 'SH600029', 'SH600030', 'SH600031', 'SH600036', 'SH600038', 'SH600048', 'SH600050', 'SH600061', 'SH600066']
>>>
```

#### 问题汇总

问题一：下载时经常卡住，一小时后仍然没有变化，命令行中显示如下：

```bash
2022-06-25 19:23:03.225 | WARNING  | qlib.tests.data:_download_data:57 - The data for the example is collected from Yahoo Finance. Please be aware that the quality of the data might not be perfect. (You can refer to the original data source: https://finance.yahoo.com/lookup.)
2022-06-25 19:23:03.227 | INFO     | qlib.tests.data:_download_data:59 - qlib_data_cn_1d_latest.zip downloading......
  9%|██████████▉                                                                                                                 | 17252352/196549189 [01:20<09:59, 298935.38it/s] 
```

原因：可能是我的win10系统休眠功能出现问题。之前曾出现类似情况，电脑进入休眠模式后无法正常恢复工作。

解决方案：使用管理员权限powershell下载，并取消系统休眠功能。

### 2.transformer模型实现

​		本地模型主要通过qlib库中自带的transformer_ts改进得到，我添加了一些中文注释并修改了部分代码段，删去了一些多余的参数，修改后代码如下：

```python
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
import math
from qlib.utils import get_or_create_path
from qlib.log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

##### 供Qlib的封装 #####
class CustomTransformerModel(Model) :
    '''
    复现Transformer模型
    1.模型相关参数：
    :param batch_size:  一个batch的长，即input的大小    
    :param d_model:     embedding 维度
    :param num_layers:  block的数量（即原论文中的Nx值）
    :param num_heads:   注意力机制有几头
    :param device:      设备名（判断是否使用gpu）
    2.训练相关参数：
    :param batch_size:  一个batch的大小
    :param num_epochs:  训练epochs次数上限
    :param n_jobs:      dataloader的并发数量
    :param early_stop:  限制过早结束训练
    '''
    def __init__(
        self,
        # 模型相关参数
        d_feat:int=20, d_model:int=64, num_layers:int=2, num_heads:int=2, dropout:float=0,
        # 训练相关参数
        batch_size:int=8192, num_epochs=100, n_jobs=4, early_stop=5,   
        # 设备参数
        GPU = 0,        # 我只有一个GPU
        # numpy参数
        lr=0.0001, reg=1e-3,
        **kwargs
        ) :
        # 初始化参数设置
        self.d_feat = d_feat
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.fitted = False     # 标志位
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.n_jobs = n_jobs
        self.early_stop = early_stop
        self.lr = lr
        self.reg = reg
        # 初始化模型
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.model = CustomTransformer(
            d_feat=self.d_feat, 
            d_model=self.d_model, 
            num_layers=self.num_layers, 
            num_heads=self.num_heads, 
            dropout=self.dropout
            )
        self.model.to(self.device)  # 模型转移给对应设备
        # 初始化日志模块
        self.logger = get_module_logger("CustomTransformerModel")
        self.logger.info("Custom Transformer:" "\nbatch_size : {}" "\ndevice : {}".format(self.batch_size, self.device))
        # optimizer
        self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)

    @property
    def use_gpu(self) :
    # 检查是当前设备是否不是cpu
    # 如果设备是gpu，返回True
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        return self.mse(pred[mask], label[mask])

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        return -self.loss_fn(pred[mask], label[mask])

    def train_epoch(self, data_loader):

        self.model.train()

        for data in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            pred = self.model(feature.float())  # .float()
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):

        self.model.eval()

        scores = []
        losses = []

        for data in data_loader:

            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float())  # .float()
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):

        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        train_loader = DataLoader(
            dl_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs, drop_last=True
        )
        valid_loader = DataLoader(
            dl_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs, drop_last=True
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.num_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())

##### transformer模型的实现 #####
### embedding ###
# 使用nn.Linear实现
### position encoder ###
class PositionalEncoder(nn.Module) :
    def __init__(self, d_model, max_len=1000):
        # 调用父组件构造函数
        super().__init__()
        # 构造位置函数
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)    # 偶数位置用正弦
        pe[:, 1::2] = torch.cos(position * div_term)    # 奇数位置用余弦

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, input) :
        return input + self.pe[: input.size(0), :]      # 输出残差
### multiHeadAttention ###
# 集成至nn.TransformerEncoderLayer中实现
### encoder ###
# 使用nn.TransformerEncoderLayer与nn.TransformerEncoder实现
### decoder ###
# 使用nn.Linear实现

### transformer模型 ###
class CustomTransformer(nn.Module):
    """
    复现的transformer模型   
    :param batch_size: 一个batch的长，即input的大小    
    :param d_model: embedding 维度
    :param num_layers:  block的数量（论文中的Nx值）
    :param num_heads:  注意力机制有几头
    :param device: 设备名（判断是否使用gpu）
    """
    # 构造
    def __init__(self, d_feat=6, d_model=8, num_layers=2, num_heads=4, dropout=0.5, device='cpu') :
        # 调用父类初始化
        super().__init__()
        # 初始化模型参数
        self.d_feat = d_feat
        self.device = device
        # 构造各部件
        self.embedding = nn.Linear(d_feat, d_model)
        self.positionEncoder = PositionalEncoder(d_model)
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoderLayer, num_layers=num_layers)
        self.decoderLayer = nn.Linear(d_model, 1)
    # 前向传播函数
    def forward(self, src):
        # 编码
        src = self.embedding(src)
        src = src.transpose(1,0)
        # 位置编码
        src = self.positionEncoder(src)
        # 编码器
        output = self.encoder(src, None)
        # 解码器
        output = self.decoderLayer(output.transpose(1,0)[:, -1, :])
        # 输出
        return output.squeeze()



```

#### 问题汇总

问题一：workflow中无法识别到自定义模型

```bash
the 'package' argument is required to perform a relative import for '.custom_pytorch_transformer_ts'
```

原因：检查报错地点发现使用了importlib

```bash
File "s:\coding\vs2019\tools\python37_64\lib\site-packages\qlib\utils\__init__.py", line 305, in get_module_by_module_path
module = importlib.import_module(module_path)
```

​	再检查代码发现有以下代码

```python
isinstance(module_path, ModuleType)
```

​	因此原因可能是我自己实现的模型所在脚本名并不能被识别为module

解决方案：由于模型所在脚本并不在python的module库中，需要使用脚本位置的相对路径



### 3.训练模型

​	首先修改transformer模型对应的workflow配置文件，主要修改provider_uri的值为本地数据集路径，并修改模型参数中的n_jobs值（主要是因为默认值为20，而我本地机器只有4核）。除此外，为了加快本次训练速度，同时对数据选取的范围进行的缩减。修改后内容如下：

```yaml
qlib_init:
    provider_uri: "S:/Coding/Worspace/Transformer/Qlib/dataset"	# 该参数为本地数据集路径
    region: cn
market: &market csi300
benchmark: &benchmark SH000300
data_handler_config: &data_handler_config
    start_time: 2008-01-01
    end_time: 2020-08-01
    fit_start_time: 2008-01-01
    fit_end_time: 2014-12-31
    instruments: *market
    infer_processors:
        - class: FilterCol
          kwargs:
              fields_group: feature
              col_list: ["RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10", 
                            "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5", 
                            "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"
                        ]
        - class: RobustZScoreNorm
          kwargs:
              fields_group: feature
              clip_outlier: true
        - class: Fillna
          kwargs:
              fields_group: feature
    learn_processors:
        - class: DropnaLabel
        - class: CSRankNorm
          kwargs:
              fields_group: label
    label: ["Ref($close, -2) / Ref($close, -1) - 1"] 

port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy
        kwargs:
            signal:
                - <MODEL> 
                - <DATASET>
            topk: 50
            n_drop: 5
    backtest:
        start_time: 2017-01-01
        end_time: 2020-08-01
        account: 100000000
        benchmark: *benchmark
        exchange_kwargs:
            limit_threshold: 0.095
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 5
task:
    model:
        class: TransformerModel
        module_path: qlib.contrib.model.pytorch_transformer_ts
        kwargs:
            seed: 0
            n_jobs: 4 # 原默认值为20
    dataset:
        class: TSDatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: Alpha158
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [2008-01-01, 2014-12-31]
                valid: [2015-01-01, 2016-12-31]
                test: [2017-01-01, 2020-08-01]
            step_len: 20
    record: 
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
          kwargs: 
            model: <MODEL>
            dataset: <DATASET>
        - class: SigAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs: 
            ana_long_short: False
            ann_scaler: 252
        - class: PortAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs: 
            config: *port_analysis_config

```

#### 问题汇总

问题一：首次运行时提示页面文件太小，无法完成操作

```bash
OSError: [WinError 1455] 页面文件太小,无法完成操作
```

原因：内存页面大小不足

解决方案：重启电脑后即可解决



问题二：提示Broken pipe，线程停止

```bash
[8868:MainThread](2022-06-25 23:16:24,707) ERROR - qlib.workflow - [utils.py:41] - An exception has been raised[BrokenPipeError: [Errno 32] Broken pipe].
```

原因：检查报错前的警告内容，提示DataLoader设置超过系统CPU核心数，会影响线程并发

```bash
 This DataLoader will create 20 worker processes in total. Our suggested max number of worker in current system is 8 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.  
```

解决方案：修改workflow的yaml配置中的n_jobs的值，使其小于CPU核心数即可



问题三：无法识别到GPU设备，导致训练时间过慢，每个epoch需训练约10分钟

原因：检查代码后了解到，训练使用的设备是通过以下代码段决定的：

```python
self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
```

​	检查发现，尽管我本机上的CUDA已经安装配置完毕，但是torch.cuda.is_available()仍旧输出为False，因此怀疑是pytorch版本与CUDA不匹配。

解决方案：检查本机CUDA版本为V11.7.64，Python版本为3.7.8，前往pytorch官网https://pytorch.org/get-started/locally/查找对应pytorch版本后，重新安装pytorch

```powershell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```



### 3.训练结果

​	在Alpha158数据集上的训练结果如下：

| 评价指标          | 训练结果 | benchmark结果 |
| ----------------- | -------- | ------------- |
| IC                | 0.0242   | 0.0264±0.00   |
| ICIR              | 0.1979   | 0.2053±0.02   |
| Rank IC           | 0.0364   | 0.0407±0.00   |
| Rank ICIR         | 0.2988   | 0.3273±0.02   |
| Annualized Return | 0.1136   | 0.0273±0.02   |
| Information Ratio | 0.5987   | 0.3970±0.26   |
| Max Drawdown      | -0.3705  | -0.1101±0.02  |



## 三、总结

### 1.评价指标

​	评价指标分为两类：预测值与实际值相关性类指标和投资回报类指标。

​	预测指标包括IC, ICIR, Rank IC, Rank ICIR；投资回报指标包括年回报率，信息率，最大回撤率

### 2.训练结果

​	本次训练使用的参数均为默认参数，因此效果相对benchmark中结果稍差。

​	原始输出结果如下：

```bash
'The following are prediction results of the CustomTransformerModel model.'   
datetime   instrument  score
2017-01-03 SH600000    0.016993
           SH600008    0.018085
           SH600009    0.064907
           SH600010    0.014842
           SH600015   -0.018570
{'IC': 0.024164739234424817,
 'ICIR': 0.19790900815887236,
 'Rank IC': 0.03641114555531185,
 'Rank ICIR': 0.29880143335621523}
```

原始回测结果输出如下：

```bash
# benchmark return 				理想收益
# excess return without cost 	无手续费情况下收益率
# excess return with cost 		有手续费情况下收益率
'The following are analysis results of benchmark return(1day).'
                       risk
mean               0.000477
std                0.012295
annualized_return  0.113561
information_ratio  0.598699
max_drawdown      -0.370479
'The following are analysis results of the excess return without cost(1day).'
                       risk
mean               0.000188
std                0.004750
annualized_return  0.044653
information_ratio  0.609384
max_drawdown      -0.108706
'The following are analysis results of the excess return with cost(1day).'
                           risk
mean               8.513219e-08
std                4.748872e-03
annualized_return  2.026146e-05
information_ratio  2.765616e-04
max_drawdown      -1.665412e-01
'The following are analysis results of indicators(1day).'
     value
ffr    1.0
pa     0.0
pos    0.0
```

