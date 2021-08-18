"""
https://ieeexplore.ieee.org/document/1380068
"""
import numpy as np
from _ipso import PSO
from _elm import ELM

class IPSOELM(PSO):
    def __init__(self) -> None:
        super().__init__(max_epoch=100)
        self.best_mdl = ELM()

    def fitness_fn(self, particles):
        # Cant think of any way of veccing this
        rmse_particles = []
        for p in particles:
            weights = p[:-1,:]
            biases = p[-1:,:].flatten()

            mdl = ELM()
            mdl.train(X=self.X_train,y=self.Y_train, input_weights=weights, biases=biases) ##### with particle
            preds = mdl.predict(self.X_val)
            rmse = np.sqrt((1/len(preds))*np.sum(preds-self.Y_val)**2)
            rmse_particles.append(rmse)
        
        print(rmse_particles)
        return rmse_particles
    
    def train(self, X_tr, y_tr, X_vl, y_vl):
        print('begin train')
        self.X_train = X_tr
        self.Y_train = y_tr
        self.X_val = X_vl
        self.Y_val = y_vl

        n_hidden_layers = 1000
        n_particles = 20
        
        particles = np.random.uniform(-1,1,size=(n_particles, X_train.shape[1]+1, n_hidden_layers))
        # generate particles
        # each particle needs to be n long m and n 
        # call optimise on particles
        glob_best = self.optimize(particles=particles)

        #self.best_mdl = ELM().train() # train on weights etc
    """
    def predict(self, X):
        return self.best_mdl.predict(X)
    """

if __name__ == "__main__":
    from sklearn.datasets import load_boston as data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X, y = data(return_X_y = True)
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=123)

    model = IPSOELM()
    model.train(X_train,y_train,X_test,y_test)

    #preds = model.predict(X_test)

    """
    For the optimisation, each particle is a m*n matrix, last row is weights
    So, the rand component needs to be a matrix of dim the same as 
    """