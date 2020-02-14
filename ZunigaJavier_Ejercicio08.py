import numpy as np
import matplotlib.pyplot as plt

def loglike1(x_obs, y_obs, sigma_y_obs, betas,model):
    n_obs = len(y_obs)
    l = 0.0
    for i in range(n_obs):
        l += -0.5*(y_obs[i]-model(x_obs[i,:], betas))**2/sigma_y_obs[i]**2
    return l

def model_A(x,params):
    y = params[0]+x*params[1]+params[2]*x**2
    return y

def model_B(x,params):
    y = params[0]*(np.exp(-0.5*(x-params[1])**2/(params[2]**2)))
    return y




def corr(data_file="data_to_fit.txt",model, n_dim=1, n_iterations=20000, sigma_y=0.1):
    data = np.loadtxt(data_file)
    x_obs = data[:,0]
    y_obs = data[:, 1]
    sigma_y_obs =  data[:, 2]
    betas = np.zeros([n_iterations, n_dim+2])

    for i in range(1, n_iterations):
        current_betas = betas[i-1,:]
        next_betas = current_betas + np.random.normal(scale=0.01, size=n_dim+2)
  
        loglike1_current = loglike1(x_obs, y_obs, sigma_y_obs, current_betas,model_A)
        
        loglike1_next = loglike1(x_obs, y_obs, sigma_y_obs, next_betas,model_A)
        
        loglike2_current = loglike1(x_obs, y_obs, sigma_y_obs, current_betas,model_B)
        
        loglike2_next = loglike1(x_obs, y_obs, sigma_y_obs, next_betas,model_B)
        
        r1 = np.min([np.exp(loglike1_next - loglike1_current), 1.0])
        alpha = np.random.random()
        r2=
        if alpha < r1:
            betas[i,:] = next_betas
        else:
            betas[i,:] = current_betas
        
    betas = betas[n_iterations//2:,:]       
    #return {'betas':betas, 'x_obs':x_obs, 'y_obs':y_obs}
( corr())


n_dim = 1

betas = results['betas']
"""A partir de aquí se deben realizar lpos """

