from model.model import FCN
from model.ensemble import EnsModel
import torch
from utils.data_initializer import initialize_data_cxplain
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from kneed import KneeLocator
import matplotlib.pyplot as plt
from utils.utils import mkdir

def get_masked_data_for_CXPlain(model, gdsc_expr):
    """
    Masks the gene expressions by zeroing out one gene per row and gives predictions to these masked samples
    Params:
        model: the predictor to be explained by CXPlain
        gdsc_expr: data used to train the predictor
    Returns:
        y_pred: the predictions without the masks
        masked_outs: the predictions when masked
    """

    x_train = torch.FloatTensor(gdsc_expr.values)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for m in model.model_list: # this is an ensemble
    #     m.to(device)
    # model.to(device)
    # model.eval()

    y_pred = model(x_train.to(device)).cpu().detach().numpy()
    n_genes = x_train.shape[1]

    mask = torch.ones((n_genes, n_genes)) - torch.eye(n_genes)

    list_of_masked_outs = []
    for i, sample in tqdm(enumerate(x_train)):
        masked_sample = sample*mask
        data = torch.utils.data.TensorDataset(masked_sample) # this is a matrix of (n_genes, n_genes)
        data = torch.utils.data.DataLoader(data, batch_size=512, shuffle=False) # do it in batches because n_genes > 15k
        
        ret_val = []
        for [x] in data:
            x = x.to(device)
            ret_val.append(model(x))

        ret_val = torch.cat(ret_val, axis=0).unsqueeze(0).cpu().detach().numpy()
        list_of_masked_outs.append(ret_val)

    masked_outs = np.concatenate(list_of_masked_outs)
    return y_pred, masked_outs 


def explain(model, gdsc_expr, gdsc_dr, masked_data, test_tcga_expr):
    """
    Trains the CXPlain model and find gene contributions for the test data
    Params:
        model: the predictor to be explained
        gdsc_expr: gene expression used for training the predictor
        gdsc_dr: drug responses used for training the predictor
        masked_data: tuple containing (1) the same data as gdsc_expr
                                      (2) the model's outputs for the training data
                                      (3) the model's outputs for the masked traning data
        test_tcga_expr: test data
    Returns:
        attribution and confidence intervals for the genes in the test set
    """


    # Import tensorflow here because we do not want to import it before it is needed
    # Tensorflow is used by CXPlain
    SEED=1
    import tensorflow as tf
    tf.compat.v1.disable_v2_behavior()
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    from tensorflow.python.keras.losses import mean_squared_error as loss
    from explainer.cxplain import MLPModelBuilder, CXPlain
    
    model_builder = MLPModelBuilder(num_layers=2, num_units=512, batch_size=16, learning_rate=0.001)

    print("Fitting CXPlain model")
    expl = CXPlain(model, model_builder, None, loss, num_models=10)
    expl.fit(gdsc_expr.values, gdsc_dr.values, masked_data=masked_data)
    attr,conf = expl.explain(test_tcga_expr.values)

    return attr, conf, expl



def boxplots(savefile, meta, outputs):
    """
    Creates boxplots of the outputs of the models according to true classes
    Params:
        savefile: filename to save the image
        meta: drug responses (ground truth)
        outputs: outputs of our models 
    """

    ctg = ["Complete Response", "Partial Response", "Stable Disease", "Clinical Progressive Disease"]
    response = ['CR', 'PR', 'SD', 'CPD']
    print(meta)

    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    
    for i in range(1, 11):
        ax = axes[(i-1)//5][(i-1)%5]

        boxes = []
        for c in ctg:
            x = meta[meta == c].index
            boxes.append(outputs.loc[x][i])

        ax.boxplot(boxes)    

        for j, box in enumerate(boxes):
            y = box.values
            x = np.random.normal(j+1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.7)

        ax.set_xticklabels(response)
        ax.set_title('seed = %d'%i)

    plt.savefig(savefile)


def load_models(FLAGS, n_genes, device):
    """
    Loads the models and places them in the device
    Params:
        FLAGS: arguments set by the user
        n_genes: number of genes
        device: cpu or cuda 
    """
    model_list = []
    
    for seed in range(1, 11):
        weights = os.path.join(FLAGS.outroot, FLAGS.folder, FLAGS.drug, str(seed), 'weights')

        mod = FCN(n_genes)
        mod.load_state_dict(torch.load(weights), strict=True)
        mod.to(device)
        mod.eval()
        model_list.append(mod)

    return model_list

def main(FLAGS, model_list):
    """
    Finds the top genes for the ensemble

    Input: 
        FLAGS: arguments set by the user
        model_list: list containing the individual model of the ensemble
    """

    # load the training data
    gdsc_expr, gdsc_dr, genes, tcga = initialize_data_cxplain(FLAGS) # reseed is here
    gene_names = pd.read_csv('data/genes.csv', index_col=0)

    # find the top genes using the labeled data only
    response = pd.read_csv('data/tcga_drug_response.csv', index_col=0).loc[FLAGS.drug].dropna()
    tcga_expr = tcga.loc[response.index, genes] # just find the contributions of the labeled ones
    n_genes = len(genes)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the models (if explain-only mode)
    if model_list is None:
        model_list = load_models(FLAGS, len(genes), device)
    else:
        for i, mod in enumerate(model_list):
            mod.to(device)
            mod.eval()
    
    # create a wrapper for the ensemble
    model = EnsModel(model_list)
    model.to(device)
    model.eval()

    # normalize the data
    ss = StandardScaler(with_std=True)
    gdsc_expr = pd.DataFrame(ss.fit_transform(gdsc_expr), index=gdsc_expr.index, columns=genes)

    # plot individual outputs of our model before explaining
    boxplot_fname = os.path.join(FLAGS.outroot, FLAGS.folder, FLAGS.drug, 'classes.png')
    x_test = torch.FloatTensor(tcga_expr.values).to(device)
    y_pred = pd.DataFrame(model.predict_indiv(x_test).cpu().detach().numpy(), index=tcga_expr.index, columns=range(1, 11))
    boxplots(boxplot_fname, response, y_pred)

    # load_precalc = True
    # if load_precalc:
    #     x_train = torch.FloatTensor(gdsc_expr.values)
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #     for m in model.model_list: # this is an ensemble
    #         m.to(device)
    #         m.eval()
    #     model.to(device)
    #     model.eval()
    #     y_pred = model(x_train.to(device)).cpu().detach().numpy()
    
    #     masked_file = os.path.join(FLAGS.outroot, FLAGS.folder, FLAGS.drug, 'masked_outs2.csv')
    #     masked_outs = pd.read_csv(masked_file, index_col=0)
    #     masked_outs = np.expand_dims(masked_outs, axis=-1)

    # else:
    y_pred, masked_outs = get_masked_data_for_CXPlain(model, gdsc_expr)
    masked_data = (gdsc_expr, y_pred, masked_outs)
    attr, conf, expl = explain(model, gdsc_expr, gdsc_dr, masked_data, tcga_expr)

    names = gene_names.loc[tcga_expr.columns, 'name']
    attr = pd.DataFrame(attr, index=tcga_expr.index, columns=names)
    attr = attr.mean(axis=0).sort_values(ascending=False)
    sorted_genes = attr.index

    # Use kneedle to find the threshold
    kneedle = KneeLocator(np.arange(len(attr)), attr, curve='convex', direction='decreasing')
    thresh = kneedle.knee
    filtered_genes = attr[sorted_genes[:thresh]]
    filtered_genes = filtered_genes/filtered_genes.max()
    filtered_genes.to_csv(os.path.join(FLAGS.outroot, FLAGS.folder, FLAGS.drug, 'top_genes.csv'))

    # save the explainer
    expl_dir = os.path.join(FLAGS.outroot, FLAGS.folder, FLAGS.drug, 'explainer')
    mkdir(expl_dir)
    expl.save(expl_dir, custom_model_saver=None)