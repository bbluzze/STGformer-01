import os
import logging
import torch
import numpy as np
from collections import OrderedDict
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from Models.loss import l2_reg_loss
from Models import utils, analysis
from Models.GNN import GNNStack  # 导入 GNNStack 类

logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less

class BaseTrainer(object):
    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat=False):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)
        self.print_conf_mat = print_conf_mat
        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):
        total_batches = len(self.dataloader)
        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)

class SupervisedTrainer(BaseTrainer):
    def __init__(self, gnn_model, transformer_model, dataloader, device, loss_module, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat=False):
        super(SupervisedTrainer, self).__init__(transformer_model, dataloader, device, loss_module, optimizer, l2_reg, print_interval, console, print_conf_mat)
        self.gnn_model = gnn_model.to(self.device)

        if isinstance(loss_module, torch.nn.CrossEntropyLoss):
            self.analyzer = analysis.Analyzer(print_conf_mat=False)
        else:
            self.classification = False
        if print_conf_mat:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)

    def train_epoch(self, epoch_num=None):
        self.model.train()
        self.gnn_model.train()
        total_loss = 0
        for i_batch, sample_batched in enumerate(self.dataloader):
            inputs, adj, labels = sample_batched
            inputs, adj, labels = inputs.to(self.device), adj.to(self.device), labels.to(self.device)

            # GNN 提取空间特征
            spatial_features = self.gnn_model(inputs, adj)

            # Transformer 建模时间依赖
            temporal_features = self.model(spatial_features)

            # 分类头
            outputs = self.model.classifier(temporal_features)

            # 计算损失
            loss = self.loss_module(outputs, labels)
            if self.l2_reg is not None:
                loss += l2_reg_loss(self.model, self.l2_reg)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if i_batch % self.print_interval == 0:
                metrics = {'loss': total_loss / (i_batch + 1)}
                self.print_callback(i_batch, metrics, prefix='Train: ')

        return total_loss / len(self.dataloader)

    def evaluate(self, epoch_num=None, keep_all=True):
        self.model.eval()
        self.gnn_model.eval()
        total_loss = 0
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(self.dataloader):
                inputs, adj, labels = sample_batched
                inputs, adj, labels = inputs.to(self.device), adj.to(self.device), labels.to(self.device)

                # GNN 提取空间特征
                spatial_features = self.gnn_model(inputs, adj)

                # Transformer 建模时间依赖
                temporal_features = self.model(spatial_features)

                # 分类头
                outputs = self.model.classifier(temporal_features)

                # 计算损失
                loss = self.loss_module(outputs, labels)
                if self.l2_reg is not None:
                    loss += l2_reg_loss(self.model, self.l2_reg)

                total_loss += loss.item()

                if i_batch % self.print_interval == 0:
                    metrics = {'loss': total_loss / (i_batch + 1)}
                    self.print_callback(i_batch, metrics, prefix='Eval: ')

        return total_loss / len(self.dataloader)

def validate(val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch):
    val_evaluator.model.eval()
    aggr_metrics = val_evaluator.evaluate(epoch)
    condition = (aggr_metrics[config['key_metric']] > best_value)
    if condition:
        best_value = aggr_metrics[config['key_metric']]
        utils.save_model(os.path.join(config['save_dir'], 'model_best.pth'), epoch, val_evaluator.model)
        best_metrics = aggr_metrics.copy()

    return aggr_metrics, best_metrics, best_value

def train_runner(config, gnn_model, transformer_model, trainer, val_evaluator, path):
    # 加载和预处理数据
    data_file_path = config.get('data_file_path')
    train_label_path = config.get('train_label_path')
    test_label_path = config.get('test_label_path')
    batch_size = config.get('batch_size', 32)
    seq_length = config.get('seq_length', 30)
    threshold = config.get('threshold', 0.5)

    # 检查数据路径是否存在
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")
    if not os.path.exists(train_label_path):
        raise FileNotFoundError(f"Train label file not found: {train_label_path}")
    if not os.path.exists(test_label_path):
        raise FileNotFoundError(f"Test label file not found: {test_label_path}")

    # 加载数据和标签
    X = np.load(data_file_path, allow_pickle=True)
    y_train = load_labels(train_label_path)
    y_test = load_labels(test_label_path)

    # 检查数据和标签的形状是否匹配
    if not isinstance(X, np.ndarray):
        raise ValueError("Loaded data is not a numpy array")
    if len(y_train) + len(y_test) != X.shape[0]:
        raise ValueError("Mismatch between data and labels length")

    # 数据标准化
    scaler = StandardScaler()
    X = np.array([scaler.fit_transform(sample) for sample in X])

    # 构建邻接矩阵
    adj_matrix = construct_adjacency_matrix(X, threshold)

    # 创建数据集和数据加载器
    train_dataset = UEA_Dataset(X[:len(y_train)], y_train, adj_matrix, seq_length)
    test_dataset = UEA_Dataset(X[len(y_train):], y_test, adj_matrix, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 更新 trainer 和 val_evaluator 的 dataloader
    trainer.dataloader = train_loader
    val_evaluator.dataloader = test_loader

    epochs = config.get('epochs', 100)
    optimizer = config['optimizer']
    loss_module = config['loss_module']
    start_epoch = 0
    total_start_time = time.time()
    tensorboard_writer = SummaryWriter('summary')
    best_value = 1e16
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}
    save_best_model = utils.SaveBestModel()
    # save_best_acc_model = utils.SaveBestACCModel()

    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc='Training Epoch', leave=False):

        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                              best_value, epoch)
        save_best_model(aggr_metrics_val['loss'], epoch, transformer_model, optimizer, loss_module, path)
        # save_best_acc_model(aggr_metrics_val['accuracy'], epoch, model, optimizer, loss_module, path)

        metrics_names, metrics_values = zip(*aggr_metrics_val.items())
        metrics.append(list(metrics_values))

        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
    total_runtime = time.time() - total_start_time
    logger.info("Train Time: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
    return

def load_labels(label_path):
    """加载标签数据"""
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.isdigit():  # 只处理数字行
                labels.append(int(line))
    return np.array(labels)

# 构建邻接矩阵
def construct_adjacency_matrix(X, threshold=0.5):
    """基于皮尔逊相关系数构建邻接矩阵"""
    correlation_matrices = []
    for sample in X:
        corr_matrix = np.corrcoef(sample.T)
        correlation_matrices.append(corr_matrix)
    mean_corr = np.mean(correlation_matrices, axis=0)
    adj = (np.abs(mean_corr) > threshold).astype(np.float32)
    np.fill_diagonal(adj, 1)
    D = np.diag(adj.sum(1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    adj_normalized = D_inv_sqrt @ adj @ D_inv_sqrt
    return adj_normalized

# UEA数据集类
class UEA_Dataset(Dataset):
    def __init__(self, X, y, adj_matrix, seq_length=30):
        self.X = X
        self.y = y
        self.adj_matrix = adj_matrix
        self.seq_length = seq_length
        self.samples = []
        for i in range(len(X)):
            num_timesteps = X[i].shape[0]
            for j in range(num_timesteps - seq_length + 1):
                self.samples.append((X[i][j:j + seq_length], y[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = torch.FloatTensor(x.T).unsqueeze(1)
        adj = torch.FloatTensor(self.adj_matrix)
        return x, adj, torch.LongTensor([y])