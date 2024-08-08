import numpy as np
from scipy.integrate import odeint

def fft_burgers(x, t, nu):

    xmin, xmax = np.min(x), np.max(x)
    nx, nt = len(x), len(t)
    dx = (xmax - xmin)/nx

    kappa = 2*np.pi*np.fft.fftfreq(nx, d=dx)
    # u0 = 1/np.cosh(x)
    u0 = 1/np.cosh(4*x-5)

    def rhsBurgers(u, t, kappa, nu):
        uhat = np.fft.fft(u)
        d_uhat = (1j)*kappa*uhat
        dd_uhat = -np.power(kappa, 2)*uhat
        d_u = np.fft.ifft(d_uhat)
        dd_u = np.fft.ifft(dd_uhat)
        du_dt = -u*d_u/4 + nu*dd_u/16
        return du_dt.real

    u = odeint(rhsBurgers, u0, t, args=(kappa, nu))

    return u
