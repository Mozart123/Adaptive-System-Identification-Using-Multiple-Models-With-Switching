#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate 3 datasets
x1 = x2 = x3 = np.linspace(0, np.pi*10, 1500)
y1 = 1 / (np.sin(x1) + 1.2) - 3
y2 = 0.5 * np.sin(x2) + 2 * np.cos(x2/2)
piece1 = np.exp(-4*np.linspace(0, np.pi/2, 75))
piece2 = -np.exp(-4*np.linspace(np.pi/2, 0, 75))
piece3 = -piece1
piece4 = -piece2
piece5 = 2*np.hstack([piece1, piece2, piece3, piece4])
y3 = np.hstack([piece5, piece5, piece5, piece5, piece5])

# Standardize data, for ANN
from sklearn.preprocessing import StandardScaler
x1_std = StandardScaler().fit_transform(np.reshape(x1, (-1, 1)))
x2_std = StandardScaler().fit_transform(np.reshape(x2, (-1, 1)))
x3_std = StandardScaler().fit_transform(np.reshape(x3, (-1, 1)))

# Show sampling rate
print(1500/(np.pi*10))


# In[2]:


# Display 3 datasets
plt.figure(figsize = (15, 4))
plt.plot(x1, y1)
plt.plot(x1, y2)
plt.plot(x1, y3)
plt.legend(['d1', 'd2', 'd3'])
plt.title('All datasets')
plt.xlabel('Time (s)')
plt.ylabel('y')

plt.figure(figsize = (15, 4))
plt.plot(x1, y1)
plt.title('d1')

plt.figure(figsize = (15, 4))
plt.plot(x1, y2)
plt.figure(figsize = (15, 4))
plt.plot(x1, y3)
plt.show()


# In[8]:


import keras
from keras.layers import *
from keras.models import Sequential
from keras import metrics
from keras.optimizers import Adam
from keras.callbacks import * 
from keras_tqdm import TQDMNotebookCallback
from keras.models import load_model
import keras.backend as K

def model1():
    model = Sequential()
    model.add(Dense(95, activation = 'relu', input_shape = (1,)))
    #model.add(BatchNormalization())
    model.add(Dense(95, activation = 'relu'))
    #model.add(BatchNormalization())
    model.add(Dense(75, activation = 'relu'))
    model.add(Dense(75, activation = 'relu'))
    #model.add(Dense(40, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))
    
    model.compile(
     optimizer = Adam(lr=0.0025, beta_1=0.95, beta_2=0.997, epsilon=None, decay=0.0, amsgrad=False),
     loss='mean_squared_error',
     metrics = [metrics.mae]
    )
    return model

class multiple_models():
    def __init__(self, hist_length, n_fixed):
        
        self.n_fixed = n_fixed
        self.models = []
        self.fixed_models = []
        self.best_model = None
        self.adaptive_model = model1()
        self.hist_length = hist_length

        self.n_models = 10#len(self.models)
        self.loss_hist = np.zeros((n_fixed, hist_length))
        self.ada_loss_hist = np.zeros(hist_length)
        self.ranking = np.arange(n_fixed)
    
    def reinitialize_fixed_model():
        self.fixed_model = model1()
    
    def get_losses(self, X, y):
        losses = np.zeros(self.n_models)
        for i,model in enumerate(self.models):
            losses[i] = model.evaluate(X,y)[0]
        return losses
    
    def train_best(self, X, y, num_to_train):
        for i_best in range(num_to_train):
            to_train = np.where(self.ranking == (i_best))[0][0]
            #print(f'Train {to_train}')
            self.models[to_train].fit(X, y,verbose = 0)
            
            
    # Save & Load Fixed Models
    def save_fixed(self):
        self.fixed_models[0].save('fixed_model0.h5')
        self.fixed_models[1].save('fixed_model1.h5')
        self.fixed_models[2].save('fixed_model2.h5')
        
    def load_fixed(self):
        self.fixed_models = []
        self.fixed_models.append(load_model('fixed_model0.h5'))
        self.fixed_models.append(load_model('fixed_model1.h5'))
        self.fixed_models.append(load_model('fixed_model2.h5'))
        
    """def pred_best(self, X, num_to_pred): # Avg preds of best n models in ranking
        pred = np.zeros(1)
        for i_best in range(num_to_pred):
            pred += (self.models[np.where(self.ranking == (i_best))[0][0]].predict(X)[0][0])/num_to_pred
        return pred"""
    
    """def collect_losses(self, X, y):
        losses = np.zeros(self.n_fixed)
        for i,model in enumerate(self.models):
            losses[i] = model.evaluate(X,y)[0]
        self.loss_hist[:, :-1] = self.loss_hist[:, 1:]
        self.loss_hist[:, -1] = losses
        return self.loss_hist.mean(axis = 1)"""
        
    def test_step(self, X, y):
        # Update loss history
        losses = np.zeros(self.n_fixed)
        for i,model in enumerate(self.fixed_models):
            losses[i] = model.evaluate(X,y)[0]
        
        loss_ada = self.adaptive_model.evaluate(X,y)[0]
        
        # Update loss hist
        self.ada_loss_hist[:-1] = self.ada_loss_hist[1:]
        self.ada_loss_hist[-1] = loss_ada
        self.loss_hist[:, :-1] = self.loss_hist[:, 1:]
        self.loss_hist[:, -1] = losses
        
        loss_means = self.loss_hist.mean(axis = 1)
        loss_mean_ada = self.ada_loss_hist.mean()
        self.best_model = np.argmin(loss_means)
        self.ranking = np.argsort(loss_means)
        best_fixed_loss = loss_means[self.best_model]
        
        if best_fixed_loss < loss_mean_ada: # SWITCH
            """self.fixed_models[self.best_model].save('model_to_switch.h5')
            self.adaptive_model = load_model('model_to_switch.h5')"""
            fixed_copy = keras.models.clone_model(self.fixed_models[self.best_model])
            fixed_copy.build((None, 1)) # replace 10 with number of variables in input layer
            fixed_copy.compile(
                                 optimizer = Adam(lr=0.006, beta_1=0.95, beta_2=0.997, epsilon=None, decay=0.0, amsgrad=False),
                                 loss='mean_squared_error',
                                 metrics = [metrics.mae]
                                )
            fixed_copy.set_weights(self.fixed_models[self.best_model].get_weights())
            
            # Change also the history
            self.ada_loss_hist = self.loss_hist[self.best_model]
        """else:
            # Train adaptive model"""
        self.adaptive_model.fit(X, y, verbose = 0)
        
        # Return best fixed model index, if adaptive is best, return -1
        if best_fixed_loss < loss_mean_ada:
            return self.best_model, best_fixed_loss, self.ada_loss_hist[-1]
        else:
            return -1, loss_mean_ada, self.ada_loss_hist[-1]
    
    def train_fixed(self, x_trn, x_val, y_trn, y_val):
        self.models.append(model1())
        #self.models.append(model1())
        self.models.append(model1())
        self.models.append(model1())
        self.models.append(model1())
        self.models.append(model1())
        self.models.append(model1())
        self.models.append(model1())
        self.models.append(model1())
        self.models.append(model1())
        
        losses = np.zeros(len(self.models))
        for i_model, model in enumerate(self.models):
            es = EarlyStopping(monitor='val_loss', mode='min', patience = 10)
            mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
            
            model.fit(x = x_trn, y = y_trn, validation_data = (x_val, y_val), callbacks = [es, mc, TQDMNotebookCallback()], batch_size = 25, verbose = 0, epochs = 150) #
            model = load_model('best_model.h5')
            
            losses[i_model] = model.evaluate(x_val, y_val)[0]
        
        best_model = self.models[np.argmin(losses)]
        best_model.save('fixed_model.h5') 
        
        # Add model to fixed models
        #K.clear_session()
        self.fixed_models.append(load_model('fixed_model.h5'))
        self.models = []
        
        return losses, np.argmin(losses)


# ## Get Fixed Models

# In[9]:


K.clear_session()
multimodel = multiple_models(hist_length = 40, n_fixed = 3)

# Train on x1_train
from sklearn.model_selection import train_test_split
X1_train, X1_val, y1_train, y1_val = train_test_split(x1_std, y1, test_size=0.33, random_state=42)
X2_train, X2_val, y2_train, y2_val = train_test_split(x2_std, y2, test_size=0.33, random_state=43)
X3_train, X3_val, y3_train, y3_val = train_test_split(x3_std, y3, test_size=0.33, random_state=44)

# Get 3 fixed models
print('Train on data1')
losses_1, best_1 = multimodel.train_fixed(X1_train, X1_val, y1_train, y1_val)
print('Train on data2')
losses_2, best_2 = multimodel.train_fixed(X2_train, X2_val, y2_train, y2_val)
print('Train on data3')
losses_3, best_3 = multimodel.train_fixed(X3_train, X3_val, y3_train, y3_val)
multimodel.save_fixed()

"""preds_1 = np.zeros(500)
preds_2 = np.zeros(500)
preds_3 = np.zeros(500)"""

preds_1 = multimodel.fixed_models[0].predict(X1_val)[:, 0]
preds_2 = multimodel.fixed_models[1].predict(X2_val)[:, 0]
preds_3 = multimodel.fixed_models[2].predict(X3_val)[:, 0]

plt.plot(losses_1)
print(best_1)
plt.plot(losses_2)
print(best_2)
plt.plot(losses_3)
print(best_3)

plt.figure(figsize = (15,5))
plt.scatter((X1_val * x1.std()) + x1.mean(), y1_val)
plt.scatter((X1_val * x1.std()) + x1.mean(), preds_1)

plt.figure(figsize = (15,5))
plt.scatter((X2_val * x1.std()) + x1.mean(), y2_val)
plt.scatter((X2_val * x1.std()) + x1.mean(), preds_2)

plt.figure(figsize = (15,5))
plt.scatter((X3_val * x1.std()) + x1.mean(), y3_val)
plt.scatter((X3_val * x1.std()) + x1.mean(), preds_3)


# In[1]:


# Training losses from fixed model selection
plt.plot(losses_1)
print(best_1)
plt.plot(losses_2)
print(best_2)
plt.plot(losses_3)
print(best_3)
plt.legend(['d1 model losses', 'd2 model losses', 'd3 model losses'])
plt.xlabel('model index')
plt.ylabel('MSE loss')


# ## Test With Switching

# In[11]:


# Load Fixed Models
K.clear_session()
multimodel = multiple_models(hist_length = 25, n_fixed = 3)
multimodel.load_fixed()


# In[12]:


# Generate Test Data
y1_test = y1 + np.random.randn(1500) * 0.4
y2_test = y2 + np.random.randn(1500) * 0.4
y3_test = y3 + np.random.randn(1500) * 0.4

X_test = x1
X_test_std = StandardScaler().fit_transform(np.reshape(X_test, (-1, 1)))
y_test = np.concatenate([y1_test[:190], y2_test[190:420], y3_test[420:600],
                         y2_test[600:890], y1_test[890:1100], y2_test[1100:1380], y3_test[1380:]], axis = 0)

# Plot Test Data
plt.figure(figsize = (15, 4))
plt.plot(x1, y1_test)
plt.plot(x1, y2_test)
plt.plot(x1, y3_test)

plt.figure(figsize = (15, 4))
plt.plot(X_test[:190], y_test[:190], c = 'r')
plt.plot(X_test[190:420], y_test[190:420], c = 'g')
plt.plot(X_test[420:600], y_test[420:600], c = 'b')
plt.plot(X_test[600:890], y_test[600:890], c = 'g')
plt.plot(X_test[890:1100], y_test[890:1100], c = 'r')
plt.plot(X_test[1100:1380], y_test[1100:1380], c = 'g')
plt.plot(X_test[1380:], y_test[1380:], c = 'b')
plt.legend(['from d1', 'from d2', 'from d3'])
plt.xlabel('Time (s)')
plt.ylabel('y')


# In[13]:


# Testing (Train Adaptive Model And Use Switching)
from IPython.display import clear_output

switched_i = []
losses_sw = []
last_switched = -1
switch_coord_y = []
switch_coord_x = []
for i_ex in range(1500):
    if i_ex % 50 == 0:
        clear_output()
        print(i_ex)
    X_test_ex = X_test_std[i_ex].reshape(1,1)
    y_test_ex = y_test[i_ex].reshape(1,1)
    i_sw, loss_mean, loss = multimodel.test_step(X_test_ex, y_test_ex)
    if i_sw == -1:
        switched_i.append(last_switched)
    else:
        switch_coord_x.append(i_ex)
        switch_coord_y.append(i_sw)
        switched_i.append(i_sw)
        last_switched = i_sw
    losses_sw.append(loss)
    
plt.figure(figsize = (15, 5))
plt.plot(switched_i)
plt.scatter(x = switch_coord_x, y = switch_coord_y)
plt.xlabel('example index')
plt.ylabel('Switched model index')

plt.figure(figsize = (15, 5))
plt.plot(losses_sw)


# In[25]:


# Display Switching Results
plt.figure(figsize = (15, 5))
plt.plot(switched_i)
plt.scatter(x = switch_coord_x, y = switch_coord_y)
plt.xlabel('example index')
plt.ylabel('Switched model index')

plt.figure(figsize = (15, 5))
plt.plot(X_test, losses_sw)

plt.figure(figsize = (15, 5))
plt.plot(X_test, np.array(switched_i) + 1)
plt.scatter(x = np.array(switch_coord_x) / 1500 * np.pi*10, y = np.array(switch_coord_y) + 1)
plt.plot(X_test[:190], y_test[:190], c = 'r')
plt.plot(X_test[190:420], y_test[190:420], c = 'g')
plt.plot(X_test[420:600], y_test[420:600], c = 'b')
plt.plot(X_test[600:890], y_test[600:890], c = 'g')
plt.plot(X_test[890:1100], y_test[890:1100], c = 'r')
plt.plot(X_test[1100:1380], y_test[1100:1380], c = 'g')
plt.plot(X_test[1380:], y_test[1380:], c = 'b')
plt.legend(['last switched index', 'from d1', 'from d2', 'from d3'])
plt.xlabel('Time (s)')
plt.ylabel('y')


# ## Compare with single adaptive model

# In[15]:


losses_ada = []
ada_model = model1()
for i_ex in range(1500):
    X_test_ex = X_test_std[i_ex].reshape(1,1)
    y_test_ex = y_test[i_ex].reshape(1,1)
    
    loss = ada_model.evaluate(X_test_ex, y_test_ex)[0]
    losses_ada.append(loss)
    
    # Train adaptive model
    ada_model.fit(X_test_ex, y_test_ex)


# In[16]:


# Plot Comparison Results
plt.figure(figsize = (15, 5))
plt.plot(losses_ada)
plt.plot(losses_sw)
plt.legend(['Only adaptive model', 'One adaptive + 3 Fixed models'])
plt.xlabel('example index')
plt.ylabel('MSE')

import pandas as pd
plt.figure()
plt.plot(pd.Series(losses_ada).rolling(25).mean())
plt.plot(pd.Series(losses_sw).rolling(25).mean())
plt.legend(['Only adaptive model', 'One adaptive + 3 Fixed models'])
plt.xlabel('example index')
plt.ylabel('E (MSE over last 25 examples)')


# In[24]:


# Plot comparison results averaged over time
import pandas as pd

plt.figure(figsize = (15, 5))
plt.plot(x1, pd.Series(losses_ada).rolling(25).mean())
plt.plot(x1, pd.Series(losses_sw).rolling(25).mean())
plt.legend(['Only adaptive model', 'One adaptive + 3 Fixed models'])
plt.xlabel('Time (s)')
plt.ylabel('E (MSE over last 25 examples)')


# In[ ]:





# In[ ]:




