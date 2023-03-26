# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:55:12 2022

@author: h5503
"""

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from itertools import product
#%%
device = torch.device('cuda:0')
#%%
df = pd.read_pickle('csi300.pkl')
#%%
def np2tensor(numpy_array):
    ts = torch.tensor(numpy_array)
    ts = ts.to(torch.float32)
    ts = ts.to(device)
    return ts

def data_loader(df):
    for i in df.groupby(level=0):
        df_date = i[1]
        X = df_date.feature.values.reshape((len(df_date), 6, -1))
        y = df_date.label.values.reshape((len(df_date), -1))
        index = df_date.index
        X, y = np2tensor(X), np2tensor(y)
        yield X, y, index

def get_A_corr(X, c=0.7):
    # X(N, T, F)
    corr = pd.DataFrame(X[:, :, 4].cpu().detach().numpy()).T.corr().values
    corr = np2tensor(corr)
    adj = (corr>c).to(torch.float32).to(device)
    return adj

def get_A_topk(X, K=5):
    # X(N, T, F)
    corr = pd.DataFrame(X[:, :, 4].cpu().detach().numpy()).T.corr().values
    corr = np2tensor(corr)
    N = corr.shape[0]
    row = torch.linspace(0, N-1, N).reshape((-1, 1)).repeat(1, K).reshape((1, -1)).long()
    column = torch.topk(corr, K, 1).indices.reshape((1, -1))
    adj = torch.zeros(N, N).to(device)
    adj[row, column] = 1
    return adj

def train_epoch(epoch, model, train_iter, loss, updater):
    model.train()
    for X, y, index in train_iter:
        y_hat = model(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.backward()
        updater.step()

def metric_fn(preds):
    precision = {}
    recall = {}
    temp = preds.groupby(level=0).apply(lambda x: x.sort_values(by='score', ascending=False))
    for k in [1, 3, 5, 10, 20, 30, 50, 100]:
        precision[k] = temp.groupby(level=0).apply(lambda x:(x.label[:k]>0).sum()/k).mean()
        recall[k] = temp.groupby(level=0).apply(lambda x:(x.label[:k]>0).sum()/(x.label>0).sum()).mean()

    ic = preds.groupby(level=0).apply(lambda x: x.label.corr(x.score)).mean()
    ic_std = preds.groupby(level=0).apply(lambda x: x.label.corr(x.score)).std()
    ir = ic/ic_std
    rank_ic = preds.groupby(level=0).apply(lambda x: x.label.corr(x.score, method='spearman')).mean()
    return precision, recall, ic, rank_ic, ir

def test_epoch(model, test_iter):
    model.eval()
    preds = []
    for X, y, index in test_iter:
        y_hat = model(X)
        preds_dict = {'score': y_hat.reshape((-1,)).cpu().detach().numpy(), 
                      'label': y.reshape((-1, )).cpu().detach().numpy()
                      }
        preds.append(pd.DataFrame(preds_dict, index=index))
    preds = pd.concat(preds, axis=0)
    precision, recall, ic, rank_ic, ir = metric_fn(preds)
    return preds, precision, recall, ic, rank_ic, ir

def train_valid_test(df_train, df_valid, df_test):
    lr = 0.01
    num_epoch = 200
    loss = nn.MSELoss()
    params = list(product([5, 10, 15], [64, 128]))
    scores = []
    models = []
    for param in params:
        model = my_model(hidden_size=param[1], K=param[0])
        updater = torch.optim.Adam(model.parameters(), lr)
        model = model.to(device)
        
        for epoch in range(num_epoch):
            train_iter = data_loader(df_train)
            train_epoch(epoch, model, train_iter, loss, updater)
        valid_iter = data_loader(df_valid)
        preds, precision, recall, ic, rank_ic, ir = test_epoch(model, valid_iter)
        scores.append(ic)
        models.append(model)
    model_select = pd.DataFrame({'param': params, 
                                 'score': scores, 
                                 'model': models})
    model_select = model_select.sort_values(by='score', ascending=False)
    best_param = model_select.param.iloc[0]
    best_model = model_select.model.iloc[0]
    
    updater = torch.optim.Adam(best_model.parameters(), lr)
    for epoch in range(num_epoch):
        train_iter = data_loader(pd.concat([df_train, df_valid]))
        train_epoch(epoch, best_model, train_iter, loss, updater)
    
    param = best_param
    model = best_model
    test_iter = data_loader(df_test)
    preds, precision, recall, ic, rank_ic, ir = test_epoch(model, test_iter)
    return preds, param
#%%
class GATLayer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        self.linear = nn.Linear(input_size, input_size)
        self.linear_e1 = nn.Linear(input_size, 1)
        self.linear_e2 = nn.Linear(input_size, 1)
        
        self.leaky_relu = nn.LeakyReLU()
        
    def forward(self, X, adj):
        X = self.linear(X)
        e = self._get_e(X)
        e = self.leaky_relu(e)
        zero = -9e15*torch.ones_like(e)
        attention = torch.where(adj>0, e, zero)
        attention = F.softmax(attention, dim=1)
        X = torch.matmul(attention, X)
        return X
        
    def _get_e(self, X):
        e1 = self.linear_e1(X)
        e2 = self.linear_e2(X)
        e = e1 + e2.T
        return e
#%%
class my_model(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, num_attentions=1, K=10):
        super().__init__()
        
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.K = K
        
        self.gru = nn.GRU(input_size=d_feat, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
                          batch_first=True)
        
        self.attentions = [GATLayer(hidden_size) for _ in range(num_attentions)]
        for i, attention in enumerate(self.attentions):
            self.add_module(f'attention_{i+1}', attention)
        
        # self.out = GATLayer(hidden_size*num_attentions, 1)
        self.out = nn.Linear(hidden_size*num_attentions, 1)
        
    def forward(self, X):
        X = X.permute(0, 2, 1)
        adj = get_A_topk(X, self.K)
        # X(N, T, F)
        X_gru, _ = self.gru(X)
        X_gru = X_gru[:, -1, :]
        
        X_gat = torch.cat([att(X_gru, adj) for att in self.attentions], dim=1)
        
        X_out = self.out(X_gat)
        return X_out
#%% 滚动窗口训练、验证并测试模型，合并所有测试结果
trade_dates = df.index.get_level_values(0).unique().sort_values()
preds = pd.DataFrame()
params = []
for i in range((len(trade_dates) - (252*4+20))//20+1):
    j = i*20
    slc_train = trade_dates[j:j+252*3]
    slc_valid = trade_dates[j+252*3:j+252*4]
    slc_test = trade_dates[j+252*4:j+252*4+20]
    preds_window, param = train_valid_test(df.loc[slc_train], df.loc[slc_valid], df.loc[slc_test])
    preds = pd.concat([preds, preds_window])
    params.append(param)
#%% 计算测试集的各项指标
trade_dates_test = preds.index.get_level_values(0).unique().sort_values()
precisions = []
recalls = []
ics = []
rank_ics = []
irs = []
for i in range(len(trade_dates_test)//20):
    j = i*20
    slc = trade_dates_test[j: j+20]
    precision, recall, ic, rank_ic, ir = metric_fn(preds.loc[slc])
    precisions.append(precision)
    recalls.append(recall)
    ics.append(ic)
    rank_ics.append(rank_ic)
    irs.append(ir)
print(f'IC mean: {np.mean(ics):.3f}, IC std: {np.std(ics)}')
print(f'IR mean: {np.mean(irs):.3f}, IR std: {np.std(irs)}')
print(f'Rank IC mean: {np.mean(rank_ics):.3f}, Rank IC std: {np.std(rank_ics)}')
#%%
metric_fn(preds)
preds.to_csv('preds_topk.csv')
preds.to_pickle('preds_topk.pkl')
