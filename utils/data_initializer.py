import pandas as pd 
import numpy as np
from utils.utils import reset_seed
import random
import torch
import os
from constants import *

def _get_splits(idx, n_folds):
	"""
	idx: list of indices
	n_folds: number of splits for n-fold
	"""
	random.shuffle(idx)
	
	folds = []
	offset = (len(idx)+n_folds)//n_folds
	for i in range(n_folds):
		folds.append(idx[i*offset:(i+1)*offset])

	return folds

def initialize_ccl(FLAGS, n_folds=5):
	"""
	Loads the CCL/GDSC dataset
	"""

	reset_seed(FLAGS.seed)
	drug = FLAGS.drug

	# load GDSC
	gdsc_expr = pd.read_csv(FLAGS.dataroot + GDSC_GENE_EXPRESSION, index_col=0).T
	gdsc_dr = pd.read_csv(FLAGS.dataroot + GDSC_lnIC50, index_col=0).T[drug].dropna()

	# normalize IC50
	genes = list(gdsc_expr.columns)
	gdsc_dr = (gdsc_dr - gdsc_dr.mean())/(gdsc_dr.values.std()) 

	idx = list(gdsc_dr.index.intersection(gdsc_expr.index))

	if n_folds > 1:
		gdsc_folds = _get_splits(idx, n_folds)
	else:
		gdsc_folds = None

	gdsc = gdsc_expr.loc[idx]
	gdsc['y'] = gdsc_dr.loc[idx]

	return gdsc, gdsc_folds, genes


def initialize_train_and_test(FLAGS):

	gdsc, gdsc_folds, genes = initialize_ccl(FLAGS)


	# Tissue informed normalization
	mean = pd.read_csv(os.path.join(FLAGS.dataroot, TCGA_MEANS), index_col=0).loc[FLAGS.drug, genes]
	std = pd.read_csv(os.path.join(FLAGS.dataroot, TCGA_STDS), index_col=0).loc[FLAGS.drug, genes]
	tcga = pd.read_csv(os.path.join(FLAGS.dataroot, TCGA_GENE_EXPRESSION), index_col=0).T
	tcga = tcga[genes]
	tcga = (tcga - mean)/std
	return gdsc, gdsc_folds, genes, tcga

def initialize_data_cxplain(FLAGS):

	reset_seed(FLAGS.seed)
	drug = FLAGS.drug

	# load GDSC
	gdsc_expr = pd.read_csv(FLAGS.dataroot + GDSC_GENE_EXPRESSION, index_col=0).T
	gdsc_dr = pd.read_csv(FLAGS.dataroot + GDSC_lnIC50, index_col=0).T[drug].dropna()

	# normalize IC50
	genes = list(gdsc_expr.columns)
	gdsc_dr = (gdsc_dr - gdsc_dr.mean())/(gdsc_dr.values.std()) 

	idx = list(gdsc_dr.index.intersection(gdsc_expr.index))

	gdsc_expr = gdsc_expr.loc[idx]
	gdsc_dr = gdsc_dr.loc[idx]

	# Tissue informed normalization
	mean = pd.read_csv(os.path.join(FLAGS.dataroot, TCGA_MEANS), index_col=0).loc[FLAGS.drug, genes]
	std = pd.read_csv(os.path.join(FLAGS.dataroot, TCGA_STDS), index_col=0).loc[FLAGS.drug, genes]
	tcga = pd.read_csv(os.path.join(FLAGS.dataroot, TCGA_GENE_EXPRESSION), index_col=0).T
	tcga = tcga[genes]
	tcga = (tcga - mean)/std

	return gdsc_expr, gdsc_dr, genes, tcga