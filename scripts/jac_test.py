import numpy as np

def norm(v, axis=None):
    return np.sqrt(np.sum(v*v, axis=axis))

def calc_jac(fun, x, args=None):

    y = fun(x) if args is None else fun(x, *args)
    if (isinstance(x, float)): cols = 1
    else: cols = x.shape[0]
    if (isinstance(y, float)): rows = 1
    else: rows = y.shape[0]

    I = np.eye(cols)*1e-6
    JFD = np.zeros((rows, cols))
    for i in range(cols):
        xp = x + I[:,i,None]
        xm = x - I[:, i, None]
        yp = fun(xp) if args is None else fun(xp, *args)
        ym = fun(xm) if args is None else fun(xm, *args)
        JFD[:,i,None] = (yp - ym)/(2*1e-6)
    return JFD

def cos_norm_delta_over_2(delta):
    return np.cos(norm(delta/2.0))

def exp_vector(delta):
    return np.sin(norm(delta/2.0)) * delta / norm(delta)

def logp1(w, xyz):
    return 2 * np.arctan2(norm(xyz), w) * xyz / norm(xyz)

def logp2(xyz, w):
    return 2 * np.arctan2(norm(xyz), w) * xyz / norm(xyz)

def invotimes(q2, q1):
    q1w = q1[0, 0]
    q1x = q1[1, 0]
    q1y = q1[2, 0]
    q1z = q1[3, 0]
    q2w = q2[0, 0]
    q2x = q2[1, 0]
    q2y = q2[2, 0]
    q2z = q2[3, 0]

    return np.array([[q2w*q1w + q2x*q1x + q2y*q1y + q2z*q1z],
                     [q2w*q1x - q2x*q1w - q2y*q1z + q2z*q1y],
                     [q2w*q1y + q2x*q1z - q2y*q1w - q2z*q1x],
                     [q2w*q1z - q2x*q1y + q2y*q1x - q2z*q1w]])

def boxminus(q2, q1):
    qtilde = invotimes(q2, q1)
    w = qtilde[0,0]
    xyz = qtilde[1:,:]
    nxyz = norm(xyz)
    return 2.0 * np.arctan2(nxyz, w) * xyz / nxyz

if __name__ == '__main__':

    print(r"d/ddelta cos (\norm{\delta / 2}) = ")
    x = np.random.randn(3,1)
    JA = -np.sin(norm(x/2.0)) * 0.5/norm(x) * x.T
    JFD = calc_jac(cos_norm_delta_over_2, x)
    print(JA)
    print(JFD)

    print(r"d/ddelta sin (\norm{\delta / 2}) * \norm{\delta} / (2 * delta) = ")
    x = np.random.randn(3,1)
    JA = np.cos(norm(x/2.0)) * 0.5/norm(x) * x * x.T/norm(x)  + np.sin(norm(x/2.0)) * (np.eye(3) * norm(x)**2 - x * x.T)/norm(x)**3
    JFD = calc_jac(exp_vector, x)
    print(JA)
    print(JFD)

    quat = np.random.randn(4,1)
    quat /= norm(quat)
    w = quat[0,0]
    xyz = quat[1:,:]
    print("log p1: d/dqw atan2")
    nxyz = norm(xyz)
    JA = - (2 * nxyz) / (nxyz**2 + w**2) * xyz / nxyz
    JFD = calc_jac(logp1, w, [xyz])
    print(JA)
    print(JFD)
    print(JA - JFD)

    print("log p2: d/dqxyz atan2")
    JA2 = (2 * w) / (nxyz**2 + w**2) * (xyz * xyz.T) / nxyz \
         + 2 * np.arctan2(nxyz, w) * ((np.eye(3) * nxyz**2 - xyz.dot(xyz.T))/nxyz**3)
    JFD = calc_jac(logp2, xyz, [w])
    print(JA2)
    print(JFD)
    print(JA2 - JFD)

    print("d/dq2 invotimes")
    q2 = np.random.randn(4, 1)
    q2 /= norm(q2)
    q1w = quat[0, 0]
    q1x = -quat[1, 0]
    q1y = -quat[2, 0]
    q1z = -quat[3, 0]
    Qmat = np.array([[q1w,  -q1x, -q1y, -q1z],
                     [-q1x, -q1w,  q1z, -q1y],
                     [-q1y, -q1z, -q1w,  q1x],
                     [-q1z,  q1y, -q1x, -q1w]])
    JFD = calc_jac(invotimes, q2, [quat])
    print(JA)
    print(JFD)
    print(Qmat - JFD)

    print("big cahuna d/dq2 boxminus")
    qtilde = invotimes(q2, quat)
    w = qtilde[0,0]
    xyz = qtilde[1:,:]

    B1 =- (2 * nxyz) / (nxyz**2 + w**2) * xyz / nxyz
    B2 = (2 * w) / (nxyz**2 + w**2) * (xyz .dot(xyz.T)) / nxyz \
         + 2 * np.arctan2(nxyz, w) * ((np.eye(3) * nxyz**2 - xyz.dot(xyz.T))/nxyz**3)
    JA = np.hstack((B1, B2)).dot(Qmat)
    JFD = calc_jac(boxminus, q2, [quat])
    print(JA)
    print(JFD)





