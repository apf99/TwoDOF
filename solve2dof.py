import numpy as np
from math import sin
from scipy.linalg import eigh
from numpy.linalg import inv
from matplotlib import pyplot as plt

def F(t):
	F = np.zeros(4)
	F[1] = F0 * np.sin(omega*t)
	return F

def G(y,t): 
	return A_inv.dot( F(t) - B.dot(y) )

def RK4_step(y, t, dt):
	k1 = G(y,t)
	k2 = G(y+0.5*k1*dt, t+0.5*dt)
	k3 = G(y+0.5*k2*dt, t+0.5*dt)
	k4 = G(y+k3*dt, t+dt)

	return dt * (k1 + 2*k2 + 2*k3 + k4) / 6

# setup the parameters
m1, m2 = 2.0, 1.0
k1, k2 = 3.0, 2.0

delta_t = 0.1
time = np.arange(0.0, 80.0, delta_t)

F0 = 0.0
omega = 1.0

y = np.array([0, 0, 0, 1])
dof = 2

# setup matrices
K = np.array([[k1+k2, -k2],[-k2, k2]])
M = np.array([[m1, 0],[0, m2]])
I = np.identity(dof)

A = np.zeros((2*dof,2*dof))
B = np.zeros((2*dof,2*dof))

A[0:2,0:2] = M
A[2:4,2:4] = I

B[0:2,2:4] = K
B[2:4,0:2] = -I

# find natural frequencies and mode shapes
evals, evecs = eigh(K,M)
frequencies = np.sqrt(evals)
print frequencies
print evecs

A_inv = inv(A)

force = []
X1 = []
X2 = []
# numerically integrate the EOMs
for t in time:
	y = y + RK4_step(y, t, delta_t) 

	X1.append(y[2])
	X2.append(y[3])
	force.append(F(t)[1])


# plot results
plt.plot(time,X1)
plt.plot(time,X2)
plt.plot(time,force)
plt.grid(True)
plt.xlabel('time (s)')
plt.ylabel('displacement (m)')
plt.title('Response Curves')
plt.legend(['X1', 'X2', 'Force'], loc='lower right')
plt.show()
