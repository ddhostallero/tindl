import numpy as np
import os
import time
from utils.data_initializer import initialize_ccl
from sklearn.preprocessing import StandardScaler
from train.trainer import Trainer
from utils.utils import setup_logger, mkdir, reset_seed
from torch.utils import data as th_data
import torch
import pandas as pd 

def cross_validation(FLAGS, gdsc, gdsc_folds, genes, lr, bs):
    """
    Does n-fold CV on the gdsc data using the given learning rate and batch size.

    Input:
        FLAGS: arguments set by the user
        gdsc: DataFrame that contains the GEx and label as columns and sample names as index
        gdsc_folds: 2D list containing the indices of the data split info n-folds
        genes: list of genes to be used
        lr: learning rate
        bs: batch size
    Output:
        maximum PCC and the number of epochs to get that max PCC for the given learning rate and batch size
    """


    log_prefix = 'cv/lr%.5f_bs%d'%(lr, bs) # prefix for the logger

    max_epoch = FLAGS.max_epoch
    n_folds = len(gdsc_folds)
    metrics = np.zeros((n_folds, max_epoch, 4))

    min_max_ep = max_epoch
    pcc = []
    n_genes = len(genes)

    for fold_index in range(n_folds):
        
        reset_seed(FLAGS.seed) # reset the seed number (for easier replicability)

        # get the validation data
        idx_val_gdsc = gdsc_folds[fold_index]
        val_gdsc = gdsc.loc[idx_val_gdsc]
        val_gdsc_expr = val_gdsc[genes].values
        val_gdsc_dr = val_gdsc['y'].values.reshape((-1,1))

        # get the training data
        idx_train_gdsc = gdsc.loc[~gdsc.index.isin(idx_val_gdsc)].index
        train_gdsc = gdsc.loc[idx_train_gdsc]
        train_gdsc_expr = train_gdsc[genes].values
        train_gdsc_dr = train_gdsc['y'].values.reshape((-1,1))

        # normalize the gene expression
        ss = StandardScaler()
        train_gdsc_expr = ss.fit_transform(train_gdsc_expr)
        val_gdsc_expr = ss.transform(val_gdsc_expr)

        print("train: %d \t val: %d"%(len(train_gdsc_expr), len(val_gdsc_expr)))

        # Prepare the Dataloaders
        train_ds = th_data.TensorDataset(
            torch.FloatTensor(train_gdsc_expr),
            torch.FloatTensor(train_gdsc_dr),
        )
        train_loader = th_data.DataLoader(train_ds, batch_size=bs, shuffle=True)

        val_ds = th_data.TensorDataset(
            torch.FloatTensor(val_gdsc_expr),
            torch.FloatTensor(val_gdsc_dr),
        )
        val_loader = th_data.DataLoader(val_ds, batch_size=bs, shuffle=False)

        # Prepare the logging object
        name = '_logs_' + FLAGS.drug
        mkdir(os.path.join(FLAGS.outroot, FLAGS.folder, FLAGS.drug, log_prefix))
        logfile = os.path.join(FLAGS.outroot, FLAGS.folder, FLAGS.drug, log_prefix, "logs_fold_%d.csv"%fold_index)
        logger = setup_logger(logfile, name)

        # Trainn the model
        trainer = Trainer(
            logger=logger,
            learning_rate=lr,
            n_genes=n_genes)

        val_pearson = trainer.train(train_loader, val_loader, max_epoch, 1)

        if len(val_pearson) < min_max_ep:
            min_max_ep = len(val_pearson)

        pcc.append(val_pearson)

    # get the best PCC and epochs
    pcc = np.asarray([x[:min_max_ep] for x in pcc])
    pcc = pcc.mean(axis=0)
    return pcc.max(), pcc.argmax()+1



def main(FLAGS):
    """
    Does a grid-search for hyperparameters
    Input:
        FLAGS: arguments set by the user
    Output:
        A dictionary containing the learning rate, batch size, and number of epochs
    """

    hyperparams = {}

    learning_rates = [5e-4, 1e-4, 5e-5, 1e-5]
    batch_size = [64, 128]

    gdsc, gdsc_folds, genes = initialize_ccl(FLAGS)

    best_pearson = -1

    # Grid-search
    for lr in learning_rates:
        for bs in batch_size:
            pcc, epoch = cross_validation(FLAGS, gdsc, gdsc_folds, genes, lr, bs)
            if pcc > best_pearson:
                best_pearson = pcc
                hyperparams['lr'] = lr 
                hyperparams['bs'] = bs
                hyperparams['epoch'] = epoch

    return hyperparams