import matplotlib.pyplot as plt
import numpy as np
from plotWindow import plotWindow


LOG_WIDTH = 1 + 7 + 3 + 7 + 3 + 7 + 3 + 2 + 2 + 2 + 15 + 15 + 15
def plotRawGPS(data, title="raw_gps"):
    np.set_printoptions(linewidth=150)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    t = data[:,0]
    xhat0 = data[:,1:8]
    vhat0 = data[:,8:11]
    xhatf = data[:, 11:18]
    vhatf = data[:, 18:21]
    x = data[:, 21:28]
    v = data[:, 28:31]
    dthat0 = data[:, 31:33]
    dthatf = data[:, 33:35]
    dt = data[:,35:37]
    s = data[:, 37:52]
    shatf = data[:, 52:67]
    shat0 = data[:, 67:82]

    xtitles = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
    vtitles = ['vx', 'vy', 'vz']
    dttitles = [r'$\tau$', r'$\dot{\tau}$']



    pw = plotWindow(title=title)

    f = plt.figure(dpi=150)
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.title(xtitles[i])
        plt.plot(t, xhat0[:,i], label=r"$\hat{x}_0$")
        plt.plot(t, xhatf[:,i], '--', linewidth=3, label=r"$\hat{x}_f$")
        plt.plot(t, x[:,i], label="$x$")
        plt.legend()
    pw.addPlot("Position", f)

    f = plt.figure(dpi=150)
    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.title(xtitles[i+3])
        plt.plot(t, xhat0[:,i+3], label=r"$\hat{x}_0$")
        plt.plot(t, xhatf[:,i+3], '--', linewidth=3, label=r"$\hat{x}_f$")
        plt.plot(t, x[:,i+3], label="$x$")
        plt.legend()
    pw.addPlot("Attitude", f)

    f = plt.figure(dpi=150)
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.title(vtitles[i])
        plt.plot(t, vhat0[:,i], label=r"$\hat{x}_0$")
        plt.plot(t, vhatf[:,i], '--', linewidth=3, label=r"$\hat{x}_f$")
        plt.plot(t, v[:,i], label="$x$")
        plt.legend()
    pw.addPlot("Velocity", f)

    f = plt.figure()
    for i in range(2):
        plt.subplot(2, 1, i+1)
        plt.title(dttitles[i])
        plt.plot(t, dthat0[:,i], label=r"$\hat{x}_0$")
        plt.plot(t, dthatf[:,i], '--', linewidth=3, label=r"$\hat{x}_f$")
        plt.plot(t, dt[:,i], label="$x$")
        plt.legend()
    pw.addPlot("Clock Bias", f)

    f = plt.figure()
    plt.suptitle("SwitchFactor")
    for i in range(15):
        plt.subplot(15, 1, i+1)
        plt.plot(t, shat0[:,i], "-r", label=r"$\hat{x}_0$")
        plt.plot(t, shatf[:,i], "-b", label=r"$\hat{x}_f$")
        plt.plot(t, s[:,i], "-g", label=r"x")
        plt.ylim(-0.1, 1.1)
        if i == 0:
            plt.legend()
    pw.addPlot("Switch Factor", f)


    pw.show()