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


