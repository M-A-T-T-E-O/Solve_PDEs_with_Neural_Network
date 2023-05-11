# Modules
import torch as nn
import numpy as np

# Function imported

# Dataset preprocessing

def preprocessing():

 # Define the Neumann Problem:  -Δu(x,y) = f(x) = 2*pi*pi*sin(pi*x)*sin(pi*y) for x ∈ (0,1), y ∈ (0,1)
 #                                u(0,y) = u(1,y) = u(x,0) = u(x,1) = 0  board condition in 1x1 square
 # The solution has the form:
 # u(x,y) = B + x*y*(1-x)*(1-y)*N(x,y,p)
 # Δu(x,y) =  -2*y*(1-y)*N(x,y,p) + (1-2x)*y*(1-y)*(δN(x,y,p)/δx) +
 #          (1-2x)*y*(1-y)*(δN(x,y,p)/δx) + x*y*(1-x)*(1-y)*(δδN(x,y,p)/δx^2) +
 #          -2*x*(1-x)*N(x,y,p) + (1-2y)*x*(1-x)*(δN(x,y,p)/δy) +
 #          (1-2y)*x*(1-x)*(δN(x,y,p)/δy) + y*x*(1-y)*(1-x)*(δδN(x,y,p)/δy^2)

 # Define True Solution u(x,y) (found analitically)
 def u(x,y):
  return np.sin(np.pi*x)*np.sin(np.pi*y)

 # Define of f(x,y)
 def f(x,y):
  return 2*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)

 # Input training set
 coordinates_train = nn.reshape(nn.from_numpy(np.linspace(0.025, 0.975, 39)).float(),(1,39))
 mesh_train = np.array(np.meshgrid(coordinates_train, coordinates_train))
 x_train = nn.from_numpy(mesh_train.T.reshape(-1, 2)).float()

 # Output training set (target for -Δu(x,y))
 y_train = f(x_train[:,0],x_train[:,1]).reshape(-1,1)

 # Input test set (thicker than the choice made for the training set)
 coordinates_test = nn.reshape(nn.from_numpy(np.linspace(0.01, 0.99, 99)).float(),(1,99))
 mesh_test = np.array(np.meshgrid(coordinates_test, coordinates_test))
 x_test = nn.from_numpy(mesh_test.T.reshape(-1, 2)).float()

 # Output test set (target for u(x,y))
 y_test = u(x_test[:,0],x_test[:,1]).reshape(-1,1)

 return x_train, y_train, x_test, y_test


