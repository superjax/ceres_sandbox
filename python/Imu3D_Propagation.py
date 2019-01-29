import matplotlib.pyplot as plt
import numpy as np

from plotWindow import plotWindow

np.set_printoptions(linewidth=150)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
pw = plotWindow()

data = np.reshape(np.fromfile('/tmp/ceres_sandbox/Imu3D.CheckPropagation.log', dtype=np.float64), (-1,27))

t = data[:, 0]
xhat = data[:, 1:11]
x = data[:, 11:21]
u = data[:, 21:27]

error = x - xhat



f = plt.figure()
plt.suptitle('Position')
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(t, x[:,i], label='x')
    plt.plot(t, xhat[:,i], label=r'\hat{x}')
    if i == 0:
        plt.legend()
pw.addPlot("Position", f)

f = plt.figure()
plt.suptitle('Velocity')
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(t, x[:,i+7], label='x')
    plt.plot(t, xhat[:, i+7], label=r'$\hat{x}$')
    if i == 0:
        plt.legend()
pw.addPlot("Velocity", f)

f = plt.figure()
plt.suptitle('Attitude')
for i in range(4):
    plt.subplot(4, 1, i+1)
    plt.plot(t, x[:,i+3], label='x')
    plt.plot(t, xhat[:,i+3], label=r'\hat{x}')
    if i == 0:
        plt.legend()
pw.addPlot("Attitude", f)

f = plt.figure()
plt.suptitle('Input')
labels=['a_x', 'a_y', 'a_z', r'$\omega_x$', r'$\omega_{y}$', r'$\omega_{z}$']
for j in range(2):
    for i in range(3):
        plt.subplot(3, 2, i*2+1 + j)
        plt.plot(t, u[:,i+j*3], label=labels[i+j*3])
        plt.legend()
pw.addPlot("Input", f)

f = plt.figure()
plt.suptitle('Error')
labels=[r'$p_x$', r'$p_y$', r'$p_z$',
        r'$v_x$', r'$v_y$', r'$v_z$',
        r'$q_x$', r'$q_y$', r'$q_z$']
for j in range(3):
    for i in range(3):
        plt.subplot(3, 3, i*3+1+j)
        plt.plot(t, error[:,i+j*3], label=labels[i+j*3])
        plt.legend()

pw.addPlot("Error", f)

pw.show()