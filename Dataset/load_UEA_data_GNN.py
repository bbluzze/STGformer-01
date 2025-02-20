import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import chardet
import re
from tqdm import tqdm


class UEA_Dataset(Dataset):
    def __init__(self, data, labels, adj_matrix):
        self.X = torch.FloatTensor(data)
        self.y = torch.LongTensor(labels)
        self.adj = torch.FloatTensor(adj_matrix)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 返回形状为 [通道数, 时间步长, 节点数]
        return self.X[idx].permute(2, 0, 1), self.adj, self.y[idx]


def load_uea_dataset(data_path, label_path):
    """加载多变量时间序列数据"""

    # 编码检测
    def get_encoding(path):
        with open(path, 'rb') as f:
            return chardet.detect(f.read())['encoding']

    # 数据解析
    try:
        data_enc = get_encoding(data_path)
        with open(data_path, 'r', encoding=data_enc, errors='replace') as f:
            data_lines = f.read().split('\n')
    except Exception as e:
        raise ValueError(f"数据文件读取失败: {str(e)}")

    X, valid_indices = [], []
    pattern = re.compile(r'^(\d+)[:,](.*)$')

    for i, line in tqdm(enumerate(data_lines), desc="解析数据"):
        if not line or line.startswith('@'):
            continue

        match = pattern.match(line)
        if not match:
            print(f"跳过无效行 {i + 1}: {line[:80]}...")
            continue

        try:
            data_str = match.group(2).replace('"', '')
            # 分割时间步和变量
            time_steps = [ts.strip() for ts in data_str.split(';') if ts.strip()]
            sample = []
            for ts in time_steps:
                variables = []
                for var in ts.split(','):
                    var = re.sub(r'[^\d.Ee+-]', '', var.strip())
                    if var:
                        variables.append(float(var))
                if variables:
                    sample.append(variables)

            if sample:
                # 转换为三维数组 [样本, 时间步, 节点]
                sample_array = np.array(sample, dtype=np.float32)
                X.append(sample_array)
                valid_indices.append(i)

        except Exception as e:
            print(f"行 {i + 1} 解析错误: {str(e)}")
            continue

    if not X:
        raise ValueError("未能加载任何有效的数据样本，请检查数据文件格式。")

    # 对齐标签
    try:
        label_enc = get_encoding(label_path)
        with open(label_path, 'r', encoding=label_enc) as f:
            all_labels = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        all_labels.append(int(line))
                    except ValueError:
                        print(f"跳过无效标签行: {line}")
            y = [all_labels[i] for i in valid_indices]
    except Exception as e:
        raise ValueError(f"标签文件错误: {str(e)}")

    # 标准化处理
    max_len = max(x.shape[0] for x in X)
    num_nodes = X[0].shape[1]

    # 填充时间步并标准化
    X_pad = []
    scaler = StandardScaler()
    for x in X:
        padded = np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant')
        X_pad.append(padded)
    X_scaled = scaler.fit_transform(np.concatenate(X_pad).reshape(-1, num_nodes))
    X_final = X_scaled.reshape(len(X_pad), max_len, num_nodes)

    print(f"成功加载 {len(X_final)} 个样本，时间步长: {max_len}，节点数: {num_nodes}")
    return X_final, np.array(y)


def construct_adjacency_matrix(X, threshold=0.5):
    """构建邻接矩阵"""
    num_samples, timesteps, num_nodes = X.shape
    corr_matrices = []

    for sample in X:
        # 计算相关系数矩阵 [节点数, 节点数]
        corr_matrix = np.corrcoef(sample.T)
        corr_matrices.append(corr_matrix)

    # 平均相关系数矩阵
    mean_corr = np.mean(corr_matrices, axis=0)

    # 应用阈值并添加自环
    adj = (np.abs(mean_corr) > threshold).astype(np.float32)
    np.fill_diagonal(adj, 1)

    # 对称归一化
    D = np.diag(adj.sum(1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

    return adj_norm


# 测试集加载和处理
def load_test_dataset(test_data_path, test_label_path, adj_matrix):
    X_test, y_test = load_uea_dataset(test_data_path, test_label_path)
    test_dataset = UEA_Dataset(X_test, y_test, adj_matrix)
    return test_dataset


# 示例用法
if __name__ == "__main__":
    train_data_path = "path/to/train/data"
    train_label_path = "path/to/train/labels"
    test_data_path = "path/to/test/data"
    test_label_path = "path/to/test/labels"

    # 加载训练数据
    X_train, y_train = load_uea_dataset(train_data_path, train_label_path)
    adj_matrix = construct_adjacency_matrix(X_train)

    # 加载测试数据
    test_dataset = load_test_dataset(test_data_path, test_label_path, adj_matrix)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 验证数据形状
    sample_x, sample_adj, sample_y = next(iter(test_dataloader))
    print(f"数据形状: {sample_x.shape}")  # [批次, 通道, 时间步, 节点]
    print(f"邻接矩阵形状: {sample_adj.shape}")
    print(f"标签形状: {sample_y.shape}")