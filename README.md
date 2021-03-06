# TINDL: Deep Learning Pipeline with Tissue-informed Normalization



## Dependencies

Please refer to `requirements.txt`. Assuming you have python, you can install the dependencies using:

```
pip install -r requirements.txt
```

## Data
The preprocessed data is available [here](https://mcgill-my.sharepoint.com/:u:/g/personal/david_hostallero_mail_mcgill_ca/EYFus5ZMkWlMk72R52Fw5BsBzXfbva8ZPSICHea8tbKBlQ?e=AY7ueF)
To use this data, extract in this directory (it will create `data` directory) or specify the `--dataroot` parameter when running (see Additional parameters)

## Running the Program

To run the entire pipeline for a single drug (tamoxifen in this example)

```
python main.py --drug=tamoxifen
```

## Additional parameters
- `--dataroot`: the root directory of your data (file names for input files can me modified in `constants.py`) (default: `./data/`)
- `--outroot`: the root directory of your outputs (default: `./results/`)
- `--folder`: subdirectory you want to save your outputs (optional)
- `--mode`: the program mode. `full` means to run the entire pipeline. `custom` only runs the training and testing using hyperparameters you defined. `explain-only` only runs the explainer (assumes trained model is given). `no-explain` only runs the hyperparameter selection and the testing.
- `--drug`: the drug you want to train/explain (default: tamoxifen)
- `--seed`: the seed number for 5-fold CV (default: 1)
- `--max_epoch`: maximum number of epochs (default: 1000)
- `--lr`: learning rate (default: 1e-4)
- `--batch_size`: size of minibatches for training (default: 128)
- `--ensemble`: number of models in the ensemble (default: 10)

## Trained weights
The trained weights are available [here](https://mcgill-my.sharepoint.com/:u:/g/personal/david_hostallero_mail_mcgill_ca/EVqS_zgObtlIltAVuym38K0BFN40SSKvel9YYw817SU_tA?e=j9bu1U)

## Disclaimer
The `explainer/cxplain/` folder is a modification of the [CXPlain](https://github.com/d909b/cxplain) repository and most of the code are directly lifted from the original repository.
