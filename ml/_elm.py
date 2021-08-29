"""
https://ieeexplore.ieee.org/document/1380068
"""
import numpy as np

class ELM():
    def __init__(self, hidden_layers = 5000) -> None:
        self.hidden_layers = hidden_layers
        
    @staticmethod
    def _relu(x):
        return np.maximum(x,0,x)

    def _forward(self, X):
        G = np.dot(X, self._input_weights)
        G += self._biases
        H = self._relu(G)
        return H
   
    def _init_weights(self, X_size):
        self._input_weights = np.random.uniform(-1,1,size=(X_size,self.hidden_layers))
        self._biases = np.random.uniform(-1,1,size=(self.hidden_layers))
    
    def fit(self, X, y, input_weights=None, biases=None):
       
        if (input_weights is None) and (biases is None):
            self._init_weights(X.shape[1])
        else:
            self._input_weights = input_weights
            self._biases = biases
        
        mpinv = np.linalg.pinv(self._forward(X))
        self._output_weights = np.dot(mpinv, y)

    def predict(self, X):
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

    #np.random.seed(1234)

    scaler = MinMaxScaler()
    X, y = load_boston(return_X_y = True)
    X_scaled = scaler.fit_transform(X)

    rmses=[]
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=1234)
    
    for i in range(100):
        model = ELM()
        model.fit(X_train,y_train)

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test,preds))
        rmses.append(rmse)
        print(rmse)
    
    print(np.mean(rmses))
    print(np.std(rmses))