"""
https://ieeexplore.ieee.org/document/1380068
"""
import numpy as np

class ELM():
    def __init__(self, n_hidden_layers = 10000, input_weights = None, biases=None) -> None:
        self.hidden_layers = n_hidden_layers
        self._input_weights = input_weights
        self._biases = biases
    
    
    def init_weights_biases(self, X_size):
        # Init weights and biases 
        if self._input_weights is None:
            self._input_weights = np.random.uniform(-1,1,size=(X_size,self.hidden_layers))
        if self._biases is None:
            self._biases = np.random.uniform(-1,1,size=(self.hidden_layers))

    @staticmethod
    def _relu(x):
        return np.maximum(x,0,x)

    def _forward(self, X):
        G = np.dot(X, self._input_weights)
        G += self._biases
        H = self._relu(G)
        return H
    
    def fit(self, X, y):
        X,y = np.array(X),np.array(y)

        self.init_weights_biases(X_size=X.shape[1])
        
        mpinv = np.linalg.pinv(self._forward(X))
        self._output_weights = np.dot(mpinv, y)

    def predict(self, X):
        X = np.array(X)
        out = self._forward(X)
        preds = np.dot(out, self._output_weights)
        return preds

if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn import svm
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error
    import time
    from tqdm import tqdm

    scaler = MinMaxScaler()
    X, y = load_boston(return_X_y = True)
    X_scaled = scaler.fit_transform(X)

    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=1234)
    
    hidden_layers = [10,20,30,40,50,100,200,300,400,500,750,1000,1250,1500,2500,3500,4500,5000,10000]
    time_taken = []
    hl_rmse = []
    n_trials = 50

    for hl in hidden_layers:
        tik = time.time()
        rmses=[]
        for i in tqdm(range(n_trials)):
            model = ELM(n_hidden_layers=hl)
            model.fit(X_train,y_train)

            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test,preds))
            rmses.append(rmse)
            
        tok = time.time()
        hl_rmse.append(np.mean(rmses))
        time_taken.append((tok-tik)/n_trials)
        print(f'hidden layers : {hl} RMSE,time: {hl_rmse[-1],time_taken[-1]}')

    print(hl_rmse)
    print(time_taken)