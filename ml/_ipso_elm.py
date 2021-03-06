"""
https://ieeexplore.ieee.org/document/1380068
"""
import numpy as np
from _pso import PSO
from _elm import ELM
from sklearn.model_selection import train_test_split

class IPSOELM(PSO):
    def __init__(self, max_epoch=50, n_hidden_layers=5000, n_particles = 10) -> None:
        super().__init__(max_epoch=max_epoch, verbose=False)
        self.n_hidden_layers = n_hidden_layers
        self.n_particles = n_particles
        self.best_mdl = ELM()
        self.control_rmse = 0

    def fitness_fn(self, particles):
        mse_particles = []
        for p in particles:
            # Get weights and biases from particle mtrx
            # Train ELM on weights and biases
            mdl = ELM(input_weights=p, biases=self.biases)
            mdl.fit(X=self.X_train,y=self.y_train) 
            # Get predictions
            preds = mdl.predict(self.X_val)
            # Get RMSE and add to rmse_particles list
            mse = np.sqrt(1/len(preds))*np.sum((preds-self.y_val)**2)

            mse_particles.append(mse)
        
        return mse_particles
    
    def fit(self, X, y):
        self.X_train, self.X_val, self.y_train, self.y_val =\
            train_test_split(X,y,test_size=0.3)
  
        # Range defines the initial search space
        particles = np.random.normal(loc=0,scale=np.sqrt(2/len(self.X_train[0])), size=(self.n_particles, len(self.X_train[0]), \
            self.n_hidden_layers))

        self.biases = np.random.uniform(0, 1, size=(self.n_hidden_layers))

        # Get random particle from swarm
        rand_idx = np.random.randint(low=0,high=self.n_particles)
        
        # Get model without any PSO
        ctrl = ELM(input_weights=particles[rand_idx],biases=self.biases)
        ctrl.fit(self.X_train, self.y_train)
        ctrl_preds = ctrl.predict(X=self.X_val)
        self.control_rmse = np.sqrt((1/len(ctrl_preds))*np.sum((ctrl_preds-self.y_val)**2))
    
        # call optimise on particles
        glob_best = self.optimize(particles=particles)
        
        # Optimised weights and biases
        self.best_mdl = ELM(n_hidden_layers=self.n_hidden_layers, input_weights=glob_best, biases=self.biases)

        # Train ELM using optimised parameters
        self.best_mdl.fit(X=self.X_train, y=self.y_train) 


    def predict(self, X):
        # Used to predict on weights once mdl trained
        return self.best_mdl.predict(X)
    

if __name__ == "__main__":
    from sklearn.datasets import load_boston as data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from tqdm import tqdm

    scaler = MinMaxScaler()
    X, y = data(return_X_y = True)
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=1234)

    mses = []
    ctrl_mse = []
    max_epoch = 100
    pbar = tqdm(total=max_epoch)
    
    for i in range(100):
        model = IPSOELM(max_epoch=max_epoch, n_particles=10, n_hidden_layers=25)
        model.fit(X_train,y_train)

        preds = model.predict(X_test)

        mse = np.sqrt((1/len(preds))*np.sum((preds-y_test)**2))
        mses.append(mse)
        ctrl_mse.append(model.control_rmse)
        pbar.set_description(f'trial {i} RMSE = {mse}')
        pbar.update()
    
    pbar.close()
    print(f'Mean PSO:{np.mean(mses)}, MEAN ELM:{np.mean(ctrl_mse)}')
    print(np.std(mses))