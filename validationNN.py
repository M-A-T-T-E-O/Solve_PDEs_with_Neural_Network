import torch
import matplotlib.pyplot as plt

# Validation algorithm for the Neural Network

def validationNN(MyNN, x_test, y_test):

 x = torch.reshape(x_test[:,0],(x_test.size(0),1))
 y = torch.reshape(x_test[:,1],(x_test.size(0),1))

 u = MyNN(x_test)*x*y*(1-x)*(1-y) + 0

 # Plot both approximated and ideal u(x,y)
 fig = plt.figure(figsize=(9, 9))
 ax = fig.gca(projection='3d')
 ax.set_title('u(x,y) analitical solution VS u_N(x,y) approximated via NN', fontsize=20)
 ax.set_xlabel("x", fontsize=16)
 ax.set_ylabel("y", fontsize=16)
 ax.set_zlabel("z", fontsize=16)
 ax.scatter(x_test[:, 0], x_test[:, 1], y_test, color='red',label='u(x,y) [ideal]',alpha=0.1)
 ax.scatter(x_test[:, 0], x_test[:, 1], u.detach().numpy(), color='green',label='u(x,y) [approximated]', alpha =0.1)
 plt.legend()
 plt.show()

 # Plot the error ( u(x,y) - u_NN(x,y,p) )
 fig_err = plt.figure(figsize=(9, 9))
 ax_err = fig_err.gca(projection='3d')
 ax_err.set_title('Error', fontsize=20)
 ax_err.set_xlabel("x", fontsize=16)
 ax_err.set_ylabel("y", fontsize=16)
 ax_err.set_zlabel("z", fontsize=16)
 ax_err.scatter(x_test[:, 0], x_test[:, 1], y_test - u.detach().numpy())
 plt.show()


 # Show both the target and the output from the trained NN
 print("\nThe output of the NN (predictions) is:\n", u, "\n\n",
       'The target output (from measurements) is:',
       "\n", y_test, "\n")


