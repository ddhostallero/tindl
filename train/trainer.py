import torch
from torch.utils import data as th_data
from torch.nn import functional as F
import numpy as np 
from scipy.stats import pearsonr 
from model.model import FCN

class Trainer():

  def __init__(self, logger, learning_rate, n_genes):
    """
    Utility class for training models
    Parameters:
      logger: logger for intermediete results
      learning_rate: learning rate for training the NN
      n_genes: number of genes
    """

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.logger = logger

    print('initializing model...')
    self.model = FCN(n_genes).to(self.device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

  def predict_label(self, expression):
    """
    Utility function the takes a numpy array and outyputs the predictions of the model
    Input:
      expression: numpy array representing the data
    Output:
      predicted labels
    """

    self.model.eval()
    with torch.no_grad():
      expression = torch.FloatTensor(expression).to(self.device)
      pred_label = self.model(expression)
      return pred_label.cpu()


  def train_step(self, train_loader):
    """
    Trains the model for 1 epoch using the data in train_loader
    Input:
      expression: PyTorch DataLoader of the training data
    Output:
      loss/MSE on the last batch
    """

    self.model.train()

    for (src_expr, src_dr) in train_loader:
      src_expr, src_dr = src_expr.to(self.device), src_dr.to(self.device)
      self.optimizer.zero_grad()

      src_pred_lab = self.model(src_expr)

      loss = torch.nn.functional.mse_loss(src_pred_lab, src_dr)
      loss.backward()
      self.optimizer.step()

    return loss.item()    


  def val_step(self, data_loader):
    """
    Validates the model using the data_loader given
    Input: 
      data_loader: PyToch DataLoader of the validation data
    Output:
      label_loss: MSE of the entire dataset in the dataloader
      pcc: pearson correlation coeffient of the predictions and true labels
    """
    self.model.eval()

    with torch.no_grad():
      label_loss = 0
      y_out = []
      y_tar = []
      for (expr, dr) in data_loader:
        y_tar.append(dr.numpy())
        # bs = expr.shape[0]

        expr, dr = expr.to(self.device), dr.to(self.device)
        pred_lab = self.model(expr)
        label_loss += F.mse_loss(pred_lab, dr, reduction='sum')#*bs_src

        y_out.append(pred_lab.cpu().numpy())

      y_out = np.concatenate(y_out, axis=0).reshape(-1)
      y_tar = np.concatenate(y_tar, axis=0).reshape(-1)
      pcc = pearsonr(y_out, y_tar)[0]

      label_loss = label_loss.item()/len(data_loader.dataset)

    return label_loss, pcc


  def train(self, train_loader, val_loader, max_epoch=1000, eval_frequency=1):
    """
    Trains the model
    Input: 
      train_loader: PyTorch DataLoader for the training set
      val_loader: PyTorch DataLoader for the validation set
      max_epoch: maximum number of epochs for training
      eval_frequency: number of epochs before evaluating on the validation set
    Output:
      list of pearson correlation coeffecients (one for each time the model was evaluated)
    """

    self.logger.info('epoch,train-mse,val-mse,OF-ratio,train-pearson,val-pearson')

    best_mse = np.infty
    best_mse_count = 0
    best_mse_ratio = 0

    pearson_list = []

    for epoch in range(max_epoch):
      batch_mse = self.train_step(train_loader)

      if ((epoch+1)%eval_frequency) == 0:
        val_mse, val_pearson = self.val_step(val_loader)   # validation perf
        train_mse, train_pearson = self.val_step(train_loader) # training performance
        pearson_list.append(val_pearson)

        print("Epoch %d, Val MSE: %.4f, Val-pearson: %.4f, Train MSE: %.4f, Train-pearson: %.4f"
          %(epoch+1, val_mse, val_pearson, train_mse, train_pearson))

        self.logger.info("%d,%f,%f,%f,%f,%f" %(epoch+1,
                          train_mse, val_mse, val_mse/train_mse, train_pearson, val_pearson))

        best_mse_count += 1
        if best_mse > val_mse:
          best_mse = val_mse
          best_mse_count = 0
          best_mse_ratio = val_mse/train_mse

        # 30 epochs have passed since the last best and the ratio increased (train overfit)
        if best_mse_count > 30 and best_mse_ratio < val_mse/train_mse:
          break

    return pearson_list

  def save_model(self, filename):
    """
    Saves the model
    Input:
      filename: the name of the file to save the weights of the model
    """
    torch.save(self.model.state_dict(), filename)
