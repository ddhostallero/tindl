import numpy as np
import os
import time
from utils.data_initializer import initialize_train_and_test
from sklearn.preprocessing import StandardScaler
from train.trainer import Trainer
from utils.utils import setup_logger, mkdir
from torch.utils import data as th_data
import torch
import pandas as pd 


def train_test(FLAGS, hyperparams, gdsc, gdsc_folds, tcga, genes, seed):
    """
    Calls the training method and tests afterwards (for one initialization)

    Input:
        FLAGS: arguments set by the user
        hyperparams: dictionary containing the learning rate, batch size, and number of epochs
        gdsc: DataFrame that contains the training GEx and label as columns and sample names as index
        gdsc_folds: 2D list containing the indices of the data split info n-folds
        tcga: DataFrame that contains the normalizez test GEx as columns and sample names as index
        genes: list of genes to be used
        seed: seed for psudo-random number generation
    Output:
        The trained model and its predictions
    """


    # create directory for this seed
    seed_dir = os.path.join(FLAGS.outroot, FLAGS.folder, FLAGS.drug, str(seed)) 
    mkdir(seed_dir)

    name = '_logs_' + FLAGS.drug
    logfile = os.path.join(seed_dir, "logs.csv")
    logger = setup_logger(logfile, name)

    trainer = Trainer(
        logger=logger,
        learning_rate=hyperparams['lr'],
        n_genes=len(genes))

    train_gdsc_expr = gdsc[genes]                    # this needs to be normalized
    train_gdsc_dr = gdsc['y'].values.reshape((-1,1)) # this is already normalized

    # normalize data
    ss = StandardScaler()
    train_gdsc_expr = ss.fit_transform(train_gdsc_expr)

    # initlaize DataLoaders
    train_ds = th_data.TensorDataset(
        torch.FloatTensor(train_gdsc_expr),
        torch.FloatTensor(train_gdsc_dr),
    )
    train_loader = th_data.DataLoader(train_ds, batch_size=hyperparams['bs'], shuffle=True)

    # although not needed, this a dummy validation set is here 
    # for exact replicability of the results in the paper; minor diff
    fold_index = 4
    idx_val_gdsc = gdsc_folds[fold_index]
    val_gdsc = gdsc.loc[idx_val_gdsc]
    val_gdsc_expr = val_gdsc[genes].values
    val_gdsc_dr = val_gdsc['y'].values.reshape((-1,1))
    val_gdsc_expr = ss.transform(val_gdsc_expr)

    val_ds = th_data.TensorDataset(
        torch.FloatTensor(val_gdsc_expr),
        torch.FloatTensor(val_gdsc_dr),
    )
    val_loader = th_data.DataLoader(val_ds, batch_size=hyperparams['bs'], shuffle=False) 

    # train the model
    trainer.train(train_loader, val_loader=val_loader, max_epoch=hyperparams['epoch'], eval_frequency=1)    
    trainer.save_model(os.path.join(seed_dir, "weights"))

    # predict on the test set
    test_pred = trainer.predict_label(tcga.values)
    test_pred = test_pred.numpy()
    result = pd.DataFrame(test_pred, index=tcga.index, columns=[FLAGS.drug]).T
    result.to_csv(os.path.join(seed_dir, 'test_prediction.csv'))

    return trainer.model.cpu(), result

def main(FLAGS, hyperparams):
    """
    Trains the models for the final ensemble
    Input:
        FLAGS: arguments set by the user
        hyperparams: dictionary of hyperparameters
    Output:
        List of models that were trained using different initializations (seeds) but with the same hyperparameters and data
    """

    mod_list = []
    orig_seed = int(FLAGS.seed)
    for i in range(1, FLAGS.ensemble+1):
        FLAGS.seed = i
        gdsc, gdsc_folds, genes, tcga = initialize_train_and_test(FLAGS) # reseeding happens here
        mod, pred = train_test(FLAGS, hyperparams, gdsc, gdsc_folds, tcga, genes, i)
        mod_list.append(mod)

        if i == 1:
            pred_df = pd.DataFrame(index=range(1, FLAGS.ensemble+1), columns=pred.columns)
        pred_df.loc[i] = pred.loc[FLAGS.drug]

    # save the ensemble predictions
    out = os.path.join(FLAGS.outroot, FLAGS.folder, FLAGS.drug, "ensemble_predictions.csv")
    pred_df.mean().to_csv(out)
    FLAGS.seed = orig_seed

    return mod_list