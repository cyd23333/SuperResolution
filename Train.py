"""
SupRes Project. Part B: Network Training

Folder structure:
cydTrain.py
cydUtils.py
models/
    SRPCNN.py
    DEAFSR.py
datasets/
    data1Hz/
        ...
    dataSRP/
        ...
results/
"""
import os
import gc
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import Utils
from models.SRPCNN import SRPCNN
from models.DenseNet import DenseNet
###########################################################################################################
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)
###########################################################################################################
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='SupRes Project.\nPart B: Load Data and Train NNs')
    parser.add_argument('--sw',          type=int, default=1, help='Switch for different ways to build the network')
    parser.add_argument('--NN_type',     type=str, default='DEAFSR', help='SRPCNN, DEAFSR')
    parser.add_argument('--dataset_type',type=str, default='1Hzx20', help='SRPD100x10, 1Hzx10, SRPD50x20, 1Hzx20')
    ###### Paras for Networks ############
    parser.add_argument('--alpha',      type=int, default=20, help='Upsampling factor')
    parser.add_argument('--epochs',     type=int, default=500, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr',         type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--seed',       type=int, default=33, help='Random seed')
    parser.add_argument('--save_every', type=int, default=10, help='Save every n epochs  (model and loss figure)')
    parser.add_argument('--plot', action='store_true', help='Plot training and validation loss')
    # parser.add_argument('--test', action='store_true', help='Test model on test set')
    args = parser.parse_args()
    
    args.plot = True
    
    # PATH setting
    rootDir = os.getcwd()
    saveDir = f'{rootDir}/results/'
    expTime = time.strftime('%y%m%d-%H%M', time.localtime())    
    expName = f'{expTime}_{args.NN_type}_{args.dataset_type}/'   
    
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    if not os.path.exists(saveDir + expName):
        os.makedirs(saveDir + expName) 
        
    # Set up logging and output parameters
    file_handler = logging.FileHandler(filename = saveDir + expName + 'run.log')
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers)
    Utils.PrintArgs(logger, args)

    # Set up device
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', DEVICE)
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    torch.backends.cudnn.deterministic = True

    # Load data and create data loaders
    if args.dataset_type == '1Hzx10':
        dataset_dir = f'{rootDir}/datasets/data1Hz/'
        train_input = torch.from_numpy(np.load(dataset_dir + 'train_input_1Hzx10.npy')).float()
        train_label = torch.from_numpy(np.load(dataset_dir + 'train_label_1Hzx10.npy')).float()
        val_input   = torch.from_numpy(np.load(dataset_dir + 'val_input_1Hzx10.npy')).float()
        val_label   = torch.from_numpy(np.load(dataset_dir + 'val_label_1Hzx10.npy')).float()        
    elif args.dataset_type == 'SRPD100x10':
        dataset_dir = f'{rootDir}/datasets/dataSRP/'
        train_input = torch.from_numpy(np.load(dataset_dir + 'train_input_100x10.npy')).float()
        train_label = torch.from_numpy(np.load(dataset_dir + 'train_label_100x10.npy')).float()
        val_input   = torch.from_numpy(np.load(dataset_dir + 'val_input_100x10.npy')).float()
        val_label   = torch.from_numpy(np.load(dataset_dir + 'val_label_100x10.npy')).float()
    elif args.dataset_type == 'SRPD50x20':
        dataset_dir = f'{rootDir}/datasets/dataSRP/'
        train_input = torch.from_numpy(np.load(dataset_dir + 'train_input_50x20.npy')).float()
        train_label = torch.from_numpy(np.load(dataset_dir + 'train_label_50x20.npy')).float()
        val_input   = torch.from_numpy(np.load(dataset_dir + 'val_input_50x20.npy')).float()
        val_label   = torch.from_numpy(np.load(dataset_dir + 'val_label_50x20.npy')).float()
        pass
    elif args.dataset_type == '1Hzx20':
        dataset_dir = f'{rootDir}/datasets/data1Hz/'
        train_input = torch.from_numpy(np.load(dataset_dir + 'train_input_1Hzx20.npy')).float()
        train_label = torch.from_numpy(np.load(dataset_dir + 'train_label_1Hzx20.npy')).float()
        val_input   = torch.from_numpy(np.load(dataset_dir + 'val_input_1Hzx20.npy')).float()
        val_label   = torch.from_numpy(np.load(dataset_dir + 'val_label_1Hzx20.npy')).float()  
    else:
        raise ValueError('Unknown dataset type!')
    
    trainSet     = TensorDataset(train_input, train_label)
    train_loader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=True)
    valSet       = TensorDataset(val_input, val_label)
    val_loader   = DataLoader(valSet, batch_size=args.batch_size, shuffle=False)


    # Set up model, loss function, optimizer, and learning rate scheduler
    print('Building model...')
    if args.NN_type == 'SRPCNN':
        model = SRPCNN(alpha=args.alpha, sw=args.sw)
    elif args.NN_type == 'DEAFSR':
        model = DenseNet(layer_num=(6,12,24,16,16),
                         growth_rate=32, init_features=64,
                         in_channels=1 , middele_channels=128)
    else:
        raise ValueError('Unknown NN type!')
    model.to(DEVICE)
    print('Model built.')

    # Define loss function
    criterion = nn.MSELoss(reduction='sum')
    # criterion = nn.L1Loss(reduction='sum')
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, verbose=True)
    
    # Define early stopping
    patience = 15
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Define other training parameters
    epochStart = 1
    if args.plot:
        train_losses = []
        val_losses = []
    
    print('\n====================\nStart training...')
    
    # Outer loop for training
    for epoch in range(epochStart, args.epochs + 1):
        epochTime = time.time()
        
        # TRAIN
        model.train()
        train_loss = 0
        for batch_idx, (train_input, train_label) in enumerate(train_loader):
            train_input, train_label = train_input.to(DEVICE), train_label.to(DEVICE)
            # nnInput = train_input.clone()
            optimizer.zero_grad()
            train_output = model(train_input)
            loss = criterion(train_output, train_label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = (train_loss/len(train_loader.dataset)) *2 # 数据集的样本总数是input+label，所以loss要乘2才正确，下同。
        
        # VALIDATION
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (val_input, val_label) in enumerate(val_loader):
                val_input, val_label = val_input.to(DEVICE), val_label.to(DEVICE)
                val_output = model(val_input)
                loss = criterion(val_output, val_label)
                val_loss += loss.item()
        val_loss = (val_loss/len(val_loader.dataset)) *2
        
        scheduler.step(val_loss)
        
        # Save model and loss figure
        if epoch % args.save_every == 0:
            Utils.SaveModel(model, optimizer, scheduler, args, epoch, saveDir + expName)
            if args.plot:
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                plt.figure(figsize=(10, 5))
                plt.plot(train_losses, label='Train loss')
                plt.plot(val_losses, label='Val loss')
                plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Loss')
                plt.savefig(f'{saveDir}{expName}loss_{str(epoch)}.png')
                plt.close()
        
        # Print training information
        logger.info(f'Epoch: {epoch}/{args.epochs} | Time cost: {(time.time()-epochTime):.1f}s   ||   Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f}')
            
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('\n\n===========Early stopping!==============')
                Utils.SaveModel(model, optimizer, scheduler, args, epoch, saveDir + expName)
                break
    
    
    gc.collect()
    torch.cuda.empty_cache()
    print('Training finished!')
    print(torch.cuda.memory_summary())
    # End of main