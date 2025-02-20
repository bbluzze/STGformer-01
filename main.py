import os
import argparse
import logging
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from art import *
# Import Project Modules -----------------------------------------------------------------------------------------------
from utils import Setup, Initialization, Data_Loader, Data_Verifier, dataset_class
from Models.model import model_factory, count_parameters
from Models.optimizers import get_optimizer
from Models.loss import get_loss_module
from Models.utils import load_model
from Training import SupervisedTrainer, train_runner
from Models.net import GNNStack  # 导入 GNNStack 类

logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()
# -------------------------------------------- Input and Output --------------------------------------------------------
parser.add_argument('--data_path', default='Dataset/UEA/', choices={'Dataset/UEA/', 'Dataset/Segmentation/'},
                    help='Data path')
parser.add_argument('--output_dir', default='Results',
                    help='Root output directory. Must exist. Time-stamped directories will be created inside.')
parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
parser.add_argument('--val_ratio', type=float, default=0.2, help="Proportion of the train-set to be used as validation")
parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- Model Parameter and Hyperparameter ---------------------------------------------
parser.add_argument('--Net_Type', default=['C-T'], choices={'T', 'C-T'}, help="Network Architecture. Convolution (C)"
                                                                              "Transformers (T)")
# Transformers Parameters ------------------------------
parser.add_argument('--emb_size', type=int, default=16, help='Internal dimension of transformer embeddings')
parser.add_argument('--dim_ff', type=int, default=256, help='Dimension of dense feedforward part of transformer layer')
parser.add_argument('--num_heads', type=int, default=8, help='Number of multi-headed attention heads')
parser.add_argument('--Fix_pos_encode', choices={'tAPE', 'Learn', 'None'}, default='tAPE',
                    help='Fix Position Embedding')
parser.add_argument('--Rel_pos_encode', choices={'eRPE', 'Vector', 'None'}, default='eRPE',
                    help='Relative Position Embedding')
# Training Parameters/ Hyper-Parameters ----------------
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.01, help='Droupout regularization ratio')
parser.add_argument('--val_interval', type=int, default=2, help='Evaluate on validation every XX epochs. Must be >= 1')
parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='accuracy',
                    help='Metric used for defining best epoch')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ System --------------------------------------------------------
parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')
args = parser.parse_args()

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    config = Setup(args)  # configuration dictionary
    device = Initialization(config)
    Data_Verifier(config)  # Download the UEA and HAR datasets if they are not in the directory
    All_Results = ['Datasets', 'ConvTran']  # Use to store the accuracy of ConvTran in e.g "Result/Datasets/UEA"

    for problem in os.listdir(config['data_path']):  # for loop on the all datasets in "data_dir" directory
        config['data_dir'] = os.path.join(config['data_path'], problem)
        print(text2art(problem, font='small'))
        # ------------------------------------ Load Data ---------------------------------------------------------------
        logger.info("Loading Data ...")

        # 动态生成数据路径
        data_file_name = f"{problem}.npy"
        train_label_file_name = f"{problem}_TRAIN.ts"
        test_label_file_name = f"{problem}_TEST.ts"

        config['data_file_path'] = os.path.join(config['data_dir'], data_file_name)
        config['train_label_path'] = os.path.join(config['data_dir'], train_label_file_name)
        config['test_label_path'] = os.path.join(config['data_dir'], test_label_file_name)

        # 加载数据
        Data = Data_Loader(config)
        train_dataset = dataset_class(Data['train_data'], Data['train_label'])
        val_dataset = dataset_class(Data['val_data'], Data['val_label'])
        test_dataset = dataset_class(Data['test_data'], Data['test_label'])

        train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        # --------------------------------------------------------------------------------------------------------------
        # -------------------------------------------- Build Model -----------------------------------------------------
        dic_position_results = [config['data_dir'].split('/')[-1]]

        logger.info("Creating model ...")
        config['Data_shape'] = Data['train_data'].shape
        config['num_labels'] = int(max(Data['train_label'])) + 1

        # 获取输入数据的通道数
        input_channels = config['Data_shape'][1]

        # 创建 GNNStack 模型用于局部注意力
        gnn_model = GNNStack(
            gnn_model_type='dyGCN2d',
            num_layers=3,
            groups=2,
            pool_ratio=0.5,
            kern_size=[3, 3, 3],
            in_dim=input_channels,  # 使用输入数据的通道数
            hidden_dim=64,
            out_dim=128,
            seq_len=16,
            num_nodes=10,
            num_classes=config['num_labels']
        ).to(device)

        # 创建 Transformer 模型用于全局注意力
        transformer_model = model_factory(config).to(device)

        logger.info("GNN Model:\n{}".format(gnn_model))
        logger.info("Transformer Model:\n{}".format(transformer_model))
        logger.info(
            "Total number of parameters: {}".format(count_parameters(gnn_model) + count_parameters(transformer_model)))

        # -------------------------------------------- Model Initialization ------------------------------------
        optim_class = get_optimizer("RAdam")
        config['optimizer'] = optim_class(list(gnn_model.parameters()) + list(transformer_model.parameters()),
                                          lr=config['lr'], weight_decay=0)
        config['loss_module'] = get_loss_module()
        save_path = os.path.join(config['output_dir'], problem + 'model_{}.pth'.format('last'))
        tensorboard_writer = SummaryWriter('summary')
        gnn_model.to(device)
        transformer_model.to(device)
        # ---------------------------------------------- Training The Model ------------------------------------
        logger.info('Starting training...')
        trainer = SupervisedTrainer(gnn_model=gnn_model, transformer_model=transformer_model, dataloader=train_loader,
                                    device=device, loss_module=config['loss_module'], optimizer=config['optimizer'],
                                    l2_reg=0,
                                    print_interval=config['print_interval'], console=config['console'],
                                    print_conf_mat=False)
        val_evaluator = SupervisedTrainer(gnn_model=gnn_model, transformer_model=transformer_model,
                                          dataloader=val_loader, device=device, loss_module=config['loss_module'],
                                          optimizer=config['optimizer'], print_interval=config['print_interval'],
                                          console=config['console'],
                                          print_conf_mat=False)

        train_runner(config, gnn_model, transformer_model, trainer, val_evaluator, save_path)
        best_gnn_model, best_transformer_model, optimizer, start_epoch = load_model(gnn_model, transformer_model,
                                                                                    save_path, config['optimizer'])
        best_gnn_model.to(device)
        best_transformer_model.to(device)

        best_test_evaluator = SupervisedTrainer(gnn_model=best_gnn_model, transformer_model=best_transformer_model,
                                                dataloader=test_loader, device=device,
                                                loss_module=config['loss_module'],
                                                optimizer=config['optimizer'], print_interval=config['print_interval'],
                                                console=config['console'],
                                                print_conf_mat=True)
        best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
        print_str = 'Best Model Test Summary: '
        for k, v in best_aggr_metrics_test.items():
            print_str += '{}: {} | '.format(k, v)
        print(print_str)
        dic_position_results.append(all_metrics['total_accuracy'])
        problem_df = pd.DataFrame(dic_position_results)
        problem_df.to_csv(os.path.join(config['pred_dir'] + '/' + problem + '.csv'))

        All_Results = np.vstack((All_Results, dic_position_results))

    All_Results_df = pd.DataFrame(All_Results)
    All_Results_df.to_csv(os.path.join(config['output_dir'], 'ConvTran_Results.csv'))