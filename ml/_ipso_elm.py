"""
https://ieeexplore.ieee.org/document/1380068
"""
import numpy as np
from _ipso import PSO
from _elm import ELM

class IPSOELM(PSO):
    def __init__(self, max_epoch=10, n_hidden_layers=5000, n_particles = 20) -> None:
        super().__init__(max_epoch=max_epoch, verbose=True)
        self.n_hidden_layers = n_hidden_layers
        self.n_particles = n_particles
        self.best_mdl = ELM()

    def fitness_fn(self, particles):
        rmse_particles = []
        for p in particles:
            # Get weights and biases from particle mtrx
            weights = p[:-1,:]
            biases = p[-1:,:].flatten()
            # Train ELM on weights and biases
            mdl = ELM()
            mdl.train(X=self.X_train,y=self.Y_train, input_weights=weights, biases=biases) 
            # Get predictions
            preds = mdl.predict(self.X_val)
            # Get RMSE and add to rmse_particles list
            rmse = np.sqrt((1/len(preds))*np.sum((preds-self.Y_val)**2))

            rmse_particles.append(rmse)
        
        return rmse_particles
    
    def train(self, X_tr, y_tr, X_vl, y_vl):
        self.X_train = X_tr
        self.Y_train = y_tr
        self.X_val = X_vl
        self.Y_val = y_vl
  
        # Range defines the initial search space
        particles = np.random.uniform(-1,1,size=(self.n_particles, X_train.shape[1]+1, \
            self.n_hidden_layers))

        # call optimise on particles
        glob_best = self.optimize(particles=particles)
        
        # Optimised weights and biases
        weights = glob_best[:-1,:]
        biases = glob_best[-1:,:].flatten()

        # Train ELM using optimised parameters
        self.best_mdl.train(X=(list(X_tr)),\
            y=(list(y_tr)),input_weights=weights,biases=biases) 
    
    def predict(self, X):
        # Used to predict on weights once mdl trained
        return self.best_mdl.predict(X)
    

if __name__ == "__main__":
    from sklearn.datasets import load_boston as data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from tqdm import tqdm

    #np.random.seed(1234)

    scaler = MinMaxScaler()
    X, y = data(return_X_y = True)
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=1234)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=1234)

    rmses = []
    for i in range(100):
        model = IPSOELM(max_epoch=20, n_particles=10)
        model.train(X_train,y_train,X_val,y_val)

        preds = model.predict(X_test)

        rmse = np.sqrt((1/len(preds))*np.sum((preds-y_test)**2))
        rmses.append(rmse)
        print(f'trial {i} RMSE = {rmse}')
    
    print(np.mean(rmses))
    print(np.std(rmses))