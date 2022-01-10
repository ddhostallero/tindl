import argparse
from utils.utils import mkdir, save_flags
import os
from train import main_cv, main_train
from explainer import find_genes

def main(FLAGS):

    # Phase 1
    if FLAGS.mode in ['full', 'no-explain']:
        # find hyperparameters using 5-fold CV
        hyperparams = main_cv.main(FLAGS)

        # write the hyperparameters to a file for replicability purposes
        with open(os.path.join(FLAGS.outroot, FLAGS.folder, FLAGS.drug, "hyperparams.txt"), 'w') as f:
            f.write('learning rate: %f\n'%hyperparams['lr'])
            f.write('batch size: %d\n'%hyperparams['bs'])
            f.write('num epoch: %d\n'%hyperparams['epoch'])
    else:
        # parameters are known
        # training with the specified hyperparams or hps found by CV
        # will ignore flags if mode != full
        hyperparams = {
            'lr': FLAGS.lr,
            'bs': FLAGS.batch_size,
            'epoch': FLAGS.max_epoch
        }

    if FLAGS.mode == 'explain-only':
        # asumes that the trained models are in <FLAGS.outroot>/<FLAGS.folder>/<FLAGS.drug>/[1-10]/weights
        model_list = None
    else:
        # train using all GDSC samples and test using the TCGA samples
        model_list = main_train.main(FLAGS, hyperparams)
   
    # Phase 2
    if FLAGS.mode == 'no-explain':
        return
    else:
        # Find the top genes that are most indicative for this specific model
        find_genes.main(FLAGS, model_list)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data/", help="root directory of the data")
    parser.add_argument("--folder", default="", help="directory of the output")
    parser.add_argument("--outroot", default="results/", help="root directory of the output")
    parser.add_argument("--mode", default="full", help="[full], custom, explain-only, no-explain")
    parser.add_argument("--drug", default='etoposide', help="drug name")
    parser.add_argument("--seed", default=1, help="seed number for pseudo-random generation", type=int)
    parser.add_argument("--max_epoch", default=1000, help="maximum number of epoch", type=int)
    parser.add_argument("--lr", default=1e-4, help="learning rate (if mode is not full)", type=float)
    parser.add_argument("--batch_size", default=128, help="batch size (if mode is not full)", type=int)
    parser.add_argument("--ensemble", default=10, help="Number of models to ensemble", type=int)

    args = parser.parse_args()
    mkdir(os.path.join(args.outroot, args.folder, args.drug))
    save_flags(args)
    main(args)



