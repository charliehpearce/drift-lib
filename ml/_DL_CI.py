import torch as t 
import numpy as np
from torch import nn
from torch.optim import Adam as optim
from torch.utils.data import TensorDataset, DataLoader
import sys

"""
To Do:
Implement fit and predict functions for easy use
Add LSTM layer instead of linear layers
"""

class QuantReg(nn.Module):
    """
    Define Model
    """
    def __init__(self, input_size, hidden_size=32) -> None:
        super(QuantReg, self).__init__()
        
        # Define basic LSTM neural net
        #self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,\
        #    batch_first = True)
        self.net = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 2))
    
    def forward(self, X):
        out = self.net(X)
        return out

class CIDLM:
    """
    Confidence Interval Deep Learning Model
    Helps with 
    """

    def __init__(self, input_size, dl_model=QuantReg, p_val = 0.2) -> None:
        self.dl_model = dl_model(input_size=input_size)
        #debig
        self.preds = []
        self.p_val = p_val
    
    @staticmethod
    def _loss_fn(y_hat, y, p_val):
        """
        In theory this could give a probability density fn..

        Args:
        y_hat - tensor of shape (n,2) of two intervals
        y - true value from regression
        p_val - target p_val of model
        """
        def get_error(y_hat_q, y, q):
            e = y - y_hat_q
            return t.mean(t.max(q*e, e*(q-1)))#dim=-1

        # Calculate quantiles from p_value
        quantile1 = p_val/2
        quantile2 = 1-(p_val/2)
        #print(quantile1,quantile2)

        # calculate errors for both
        error1 = get_error(y_hat[:,0],y,quantile1)
        error2 = get_error(y_hat[:,1],y,quantile2)
        
        return (error1+error2)/2
    
    def _train(self, train_loader, n_epoch=1000, optim=optim, learning_rate=0.0001, verbose = False):
        
        optimizer = optim(self.dl_model.parameters(),learning_rate)
     
        for epoch in range(n_epoch):
            self.dl_model.train()
            losses = []
            
            for feats, labs in train_loader:
                optimizer.zero_grad()
                out = self.dl_model(feats)
                l = self._loss_fn(out, labs, self.p_val)
                losses.append(l)
                l.backward()
                optimizer.step()
            
            if verbose:
                print(f'Epoch {epoch+1} loss : {t.mean(l).detach()}')
    
    def fit(self, X, y, **kwargs):
        """
        Fit argument can pass kwargs down to train function
        """
        # Create dataloader from X and y
        X_train = X.astype(np.float32)
        y_train = y.astype(np.float32)

        train_tensorset = TensorDataset(t.tensor(X_train),t.tensor(y_train))
        train_dataset = DataLoader(train_tensorset, batch_size=50, shuffle=True)

        self._train(train_dataset, **kwargs)

    def predict(self,X):
        self.dl_model.eval()
        X_tensor = t.tensor(X.astype(np.float32))
        return self.dl_model(X_tensor).detach().numpy()


if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing as data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X, y = data(return_X_y = True)
    X_scaled = scaler.fit_transform(X)

    rmses=[]
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=1234)
    
    dl_model = QuantReg
    model = CIDLM(dl_model=dl_model, input_size=X_train.shape[1])
    model.fit(X_train,y_train, n_epoch=100)
    preds = model.predict(X_test).flatten()

    rmse = np.sqrt(np.mean((preds-y_test)**2))
    print(rmse)