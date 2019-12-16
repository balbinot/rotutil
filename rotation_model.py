#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import sin, cos, log, log10, sqrt, pi, exp
from numpy.random import randn, rand
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def logistic(x, k, b):
    return(1./(1+np.exp(-k*(x-b))))

def ilogistic(x, k, b):
    return(1./(1+np.exp(-k*(-x+b))))

class Rotation(object):
    """Axissymetric rotation utilities"""

    def __init__(self, i, PA, kind='solid', **kwargs):
        """Produce a mixed rotation model

        :i: Inclination in degrees (required)
        :PA: Position Angle in degrees (required)
        :kind: solid or rpeak (Lynden-Bell)
        :w: angular velocity
        :Rpeak: position of the peak rotation amplitude (only for kind==rpeak)

        """
        self._i = np.deg2rad(i)
        self._PA = np.deg2rad(PA)
        self._kind = kind

        if kind == 'solid':
            self._w = kwargs['w']
        elif kind == 'rpeak':
            self._w = kwargs['w']
            self._Rpeak = kwargs['Rpeak']

        # Just for completeness
        self._dtheta = 0.0
        self._dr = 0.0

    def _Vr(self):
        return self._dr

    def _Vtheta(self, r, theta):
        return r * self._dtheta

    def _Vphi(self, r, theta):
        """ phi velocity """
        if self._kind == 'solid':
            return r * sin(theta) * self._w
        elif self._kind == 'rpeak':
            x = self._xp(r, theta, 0)
            y = self._yp(r, theta, 0)
            z = self._zp(r, theta, 0)
            return self._w * np.sqrt(x**2 + y**2)/(1 +
                                                   (self._r(x,y,z)/self._Rpeak)**2)

    def _xp(self, r, theta, phi):
        """Cluster reference frame x component"""
        return r * sin(theta) * cos(phi)

    def _yp(self, r, theta, phi):
        """Cluster reference frame y component"""
        return r * sin(theta) * sin(phi)

    def _zp(self, r, theta, phi):
        """Cluster reference frame z component"""
        return r * cos(theta)

    def _vxp(self, r, theta, phi):
        """ Cluster frame x velocity """
        Vr = self._Vr()
        Vtheta = self._Vtheta(r, theta)
        Vphi = self._Vphi(r, theta)
        return (Vr * sin(theta) * cos(phi) + Vtheta * cos(theta) * cos(phi) -
                Vphi * sin(phi))

    def _vyp(self, r, theta, phi):
        """ Cluster frame x velocity """
        Vr = self._Vr()
        Vtheta = self._Vtheta(r, theta)
        Vphi = self._Vphi(r, theta)
        return (Vr * sin(theta) * sin(phi) + Vtheta * cos(theta) * sin(phi) +
                Vphi * cos(phi))

    def _vzp(self, r, theta, phi):
        """ Cluster frame x velocity """
        Vr = self._Vr()
        Vtheta = self._Vtheta(r, theta)
        return Vr * cos(theta) - Vtheta * sin(theta)

    def _r(self, x, y, z):
        """ Cluster frame r"""
        return np.sqrt(x**2 + y**2 + z**2)

    def _theta(self, x, y, z):
        """ Cluster frame theta"""
        return np.arccos(z / self._r(x, y, z))

    def _phi(self, x, y, z):
        """ Cluster frame phi"""
        return np.arctan2(y, x)

    def _vr(self, x, y, z, dx, dy, dz):
        """ Vr from cartesian components """
        return (x * dx + y * dy + z * dz) / self._r(x, y, z)

    def _vtheta(self, x, y, z, dx, dy, dz):
        """ Vtheta from cartesian components """
        return (y * dx - x * dy) / np.sqrt(x**2 + y**2)

    def _vphi(self, x, y, z, dx, dy, dz):
        """ Vphi from cartesian components """
        A = z * (x * dx + y * dy) - (x**2 + y**2) * dz
        B = np.sqrt(x**2 + y**2) * self._r(x, y, z)**2
        return A / B

    def _x(self, xp, yp, zp):
        """ On-sky x """
        PA = self._PA
        i = self._i
        x = xp * cos(PA) - yp * cos(i) * sin(PA) + zp * sin(i) * sin(PA)
        return x

    def _y(self, xp, yp, zp):
        """ On-sky y """
        PA = self._PA
        i = self._i
        y = xp * sin(PA) + yp * cos(i) * cos(PA) - zp * sin(i) * cos(PA)
        return y

    def _z(self, xp, yp, zp):
        """ On-sky z """
        i = self._i
        z = yp * sin(i) + zp * cos(i)
        return z

    def XPA(self, x, y):
        """ Returns X_PA from on-sky coordinates """
        return x * cos(self._PA) + y * sin(self._PA)

    def YPA(self, x, y):
        """ Returns Y_PA from on-sky coordinates """
        return -x * sin(self._PA) + y * cos(self._PA)

    def _D(self, x, y, z):
        """ Return the 3-D distance from the on-sky coordinates """
        X = self.XPA(x, y)
        Y = self.YPA(x, y)
        i = self._i
        return (
            X**2 + (Y * cos(i) + z * sin(i))**2 + (z * cos(i) - Y * sin(i))**2)

    def vcomp(self, vx, vy):
        """Take on-sky velocities and transform to vpar and vperp"""

        vpar = -vx*sin(PA) + vy*cos(PA)
        vper =  vx*cos(PA) + vy*sin(PA)

        return(vpar, vper)


    def vrpeak(self, x, y, z):
        """
        Input:
            x, y, z: 3D Position
        Output:
            vx, vy, vz: 3 velocity components

        Notes: return all 3 velocity components from the Rpeak rotation, this is
        the same as vsolid, but the angular velocity has an r dependecy."""

        Rpeak = self._Rpeak
        i = self._i
        PA = self._PA
        w = self._w/(1 + (self._r(x, y, z)/Rpeak)**2)

        vx = w * (-z * sin(i) * cos(PA) - y * cos(i))
        vy = w * (-z * sin(i) * sin(PA) + x * cos(i))
        vz = w * sin(i) * (x * cos(PA) + y * sin(PA))

        return (vx, vy, vz)

    def vsolid(self, x, y, z):
        """
        Input:
            x, y, z: 3D Position
        Output:
            vx, vy, vz: 3 velocity components

        Notes: return all 3 velocity components from a solid body rotation"""

        w = self._w
        PA = self._PA
        i = self._i

        vx = w * (-z * sin(i) * cos(PA) - y * cos(i))
        vy = w * (-z * sin(i) * sin(PA) + x * cos(i))
        vz = w * sin(i) * (x * cos(PA) + y * sin(PA))

        return (vx, vy, vz)

    def _g(self, A, B):
        if np.any(B/A <= 1):
            return np.arccos(B/A)/np.sqrt(A*A - B*B)
        if np.any(B/A > 1):
            return np.arccosh(B/A)/np.sqrt(B*B - A*A)
        else:
            raise ValueError("g should not be used if a = Rp")

    def _C01(self, R, w, a, Rp):
        A = np.sqrt(a**2 + R**2)
        B = np.sqrt(Rp**2 + R**2)
        if a != Rp:
            return w*Rp**2*(3*A**4 * self._g(A, B) - 5*B*A**2 + 2*B**3)/(2*B*(a**2 - Rp**2)**2)
        else:
            return 4*w*Rp**2/(5*(Rp**2 + R**2))

    def _C02(self, R, w, a, Rp):
        A = np.sqrt(a**2 + R**2)
        B = np.sqrt(Rp**2 + R**2)
        if a != Rp:
            return (w**2)*(Rp**4)*(-3*A**4*B - 16*A**2*B**3 + 4*B**5 - 3*A**4*(A**2 - 6*B**2)*self._g(A,B))/(4*B**3*(Rp**2 - a**2)**3)
        else:
            return (24./35.)*Rp**4*w**2/(Rp**2 + R**2)**2

    def _C22(self, R, w, a, Rp):
        A = np.sqrt(a**2 + R**2)
        B = np.sqrt(Rp**2 + R**2)
        if a != Rp:
            return (w**2)*(Rp**4)*(13*A**4*B + 2*A**2*B**3 - 3*A**4*(A**2 + 4*B**2)*self._g(A,B))/(4*B*(Rp**2 - a**2)**3)
        else:
            return (4./35.)*Rp**4*w**2/(Rp**2 + R**2)


    def m1v(self, x, y, a):
        """Raw first moment of the velocity, given a Plummer profile with scale radius *a*.
        Essentially it is vx, vy, vz, vpar, vper but without the z dependency. For the Rpeak
        case the angular velocity is replaced by C01 which can be seen as the "effective
        angular velocity"
        """
        w = self._w
        i = self._i
        PA = self._PA
        R = np.sqrt(x*x + y*y)
        if self._kind=="solid":
            vx = w * (-y * cos(i))
            vy = w * (x * cos(i))
            vz = w * sin(i) * (x * cos(PA) + y * sin(PA))
            vpar = w * (x * cos(PA) + y * sin(PA))*cos(i)
            vper = w * (-y * cos(PA) + x * sin(PA))*cos(i)

        if self._kind=="rpeak":
            Rp = self._Rpeak
            C01 = self._C01(R, w, a, Rp)
            vx = C01 * (-y * cos(i))
            vy = C01 * (x * cos(i))
            vz = C01 * sin(i) * (x * cos(PA) + y * sin(PA))
            vpar = C01 * (x * cos(PA) + y * sin(PA))*cos(i)
            vper = C01 * (-y * cos(PA) + x * sin(PA))*cos(i)

        return np.array([vx, vy, vz, vpar, vper])

    def m2v(self, x, y, a):
        """ Central second moment of the velocity, given a Plummer profile with scale radius *a*.
        Essentially it is vx, vy, vz, vpar, vper but without the z dependency. For the Rpeak
        case the angular velocity is replaced by C01 which can be seen as the "effective
        angular velocity"
        """
        w = self._w
        i = self._i
        PA = self._PA
        R = np.sqrt(x*x + y*y)
        if self._kind=="solid":
            vvx = 0.5 * w * w * (sin(i)**2 * cos(PA)**2 *(a*a + R*R) - y*y*cos(i)**2)
            vvy = 0.5 * w * w * (sin(i)**2 * cos(PA)**2 *(a*a + R*R) - x*x*cos(i)**2)
            vvz = np.zeros_like(x)
            vvpar = np.zeros_like(x)
            vvper = 0.5 * w * w * (a*a + R*R) * sin(i)**2 - (w*(-y * cos(PA) + x * sin(PA))*cos(i))**2

        if self._kind=="rpeak":
            Rp = self._Rpeak
            C01 = self._C02(R, w, a, Rp)
            C02 = self._C02(R, w, a, Rp)
            C22 = self._C22(R, w, a, Rp)

            # recomput first moments for convenience
            vx = C01 * (-y * cos(i))
            vy = C01 * (x * cos(i))
            vz = C01 * sin(i) * (x * cos(PA) + y * sin(PA))
            vpar = C01 * (x * cos(PA) + y * sin(PA))*cos(i)
            vper = C01 * (-y * cos(PA) + x * sin(PA))*cos(i)

            # since this is the *central* second comment, must remove the first moment squared of each component.
            vvx = C22 * sin(i)**2 * cos(PA)**2 + C02* y*y*cos(i)**2 - vx**2
            vvy = C22 * sin(i)**2 * cos(PA)**2 + C02* x*x*cos(i)**2 - vy**2
            vvz = C02 * (vz/C01)**2 - vz**2
            vvpar = C02 * (vpar/C01)**2 + C22*sin(i)*2 - vpar**2
            vvper = C02 * (vper/C01)**2 + C22*sin(i)*2 - vper**2

        return np.array([vvx, vvy, vvz, vvpar, vvper])

if __name__=='__main__':

    # Solid body
    ws = 0.015 # This gives 2km/s at 130 pc (on the large side)

    # Violent relax rotation
    Rp = 9.5            # parsec
    wp = 2*3.25/Rp      # 3.25km/s @ Rp
    Rs = 5.5            # scale radius of Plumar profile

    ## Parameters that control the stiching of the two rotation models
    K = 0.3       # how quickly the transition happens
    Rb = 130/2. # where it is centred at

    ## Initialize the models with two sets of inclination and PA (two frist parameters in the Rotation() class
    Ms = Rotation(40, 30, kind='solid', w=ws)
    Mp = Rotation(30, 60, kind='rpeak', w=wp, Rpeak=Rp)

    ## Grid where to compute the model in
    x, y = np.arange(-120, 120, 3), np.arange(-120, 120, 3)
    xx, yy = np.meshgrid(x,y)
    rr = np.sqrt(xx*xx + yy*yy)

    ## Raw 1st moment
    Vs = Ms.m1v(xx, yy, Rs)
    Vp = Mp.m1v(xx, yy, Rs)

    ## 2nd central moment
    VVs = Ms.m2v(xx, yy, Rs)
    VVp = Mp.m2v(xx, yy, Rs)

    ## Do the stiching
    V =  logistic(rr, K, Rb)*Vs + ilogistic(rr, K, Rb)*Vp
    VV = logistic(rr, K, Rb)*VVs + ilogistic(rr, K, Rb)*VVp ## <v^2> = Var = Disp^2
    VV = np.sqrt(VV)
    VV = np.nan_to_num(VV, 0)

    ## Plot all components.
    import matplotlib.gridspec as gridspec
    plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(2, 5)
    ras = True

    vmin =  None
    vmax =  None
    vminr = None
    vmaxr = None
    ax = plt.subplot(gs[0, 0])
    img = ax.pcolor(xx, yy, V[0], vmin=vmin, vmax=vmax, rasterized=ras)
    ax.set_ylabel('y [pc]')
    ax.set_xlabel('x [pc]')
    cb = plt.colorbar(img)
    cb.set_label('vx')
    cb.set_label(r'$\langle v_x \rangle$ [km/s]')

    ax = plt.subplot(gs[0, 1])
    img = ax.pcolor(xx, yy, V[1], vmin=vmin, vmax=vmax, rasterized=ras)
    ax.set_ylabel('y [pc]')
    ax.set_xlabel('x [pc]')
    cb = plt.colorbar(img)
    cb.set_label(r'$\langle v_y \rangle$ [km/s]')

    ax = plt.subplot(gs[0, 2])
    img = ax.pcolor(xx, yy, V[2], vmin=vminr, vmax=vmaxr, rasterized=ras)
    ax.set_ylabel('y [pc]')
    ax.set_xlabel('x [pc]')
    cb = plt.colorbar(img)
    cb.set_label(r'$\langle v_z \rangle$ [km/s]')

    ax = plt.subplot(gs[0, 3])
    img = ax.pcolor(xx, yy, V[3], vmin=vminr, vmax=vmaxr, rasterized=ras)
    ax.set_ylabel('y [pc]')
    ax.set_xlabel('x [pc]')
    cb = plt.colorbar(img)
    cb.set_label(r'$\langle v_{\parallel} \rangle$ [km/s]')

    ax = plt.subplot(gs[0, 4])
    img = ax.pcolor(xx, yy, V[4], vmin=vminr, vmax=vmaxr, rasterized=ras)
    ax.set_ylabel('y [pc]')
    ax.set_xlabel('x [pc]')
    cb = plt.colorbar(img)
    cb.set_label(r'$\langle v_{\bot} \rangle$ [km/s]')

    ax = plt.subplot(gs[1, 0])
    img = ax.pcolor(xx, yy, VV[0], vmin=vmin, vmax=vmax, rasterized=ras)
    ax.set_ylabel('y [pc]')
    ax.set_xlabel('x [pc]')
    cb = plt.colorbar(img)
    cb.set_label('vx')
    cb.set_label(r'$\langle v_x^2 \rangle$ [km/s]')

    ax = plt.subplot(gs[1, 1])
    img = ax.pcolor(xx, yy, VV[1], vmin=vmin, vmax=vmax, rasterized=ras)
    ax.set_ylabel('y [pc]')
    ax.set_xlabel('x [pc]')
    cb = plt.colorbar(img)
    cb.set_label(r'$\langle v_y^2 \rangle$ [km/s]')

    ax = plt.subplot(gs[1, 2])
    img = ax.pcolor(xx, yy, VV[2], vmin=vminr, vmax=vmaxr, rasterized=ras)
    ax.set_ylabel('y [pc]')
    ax.set_xlabel('x [pc]')
    cb = plt.colorbar(img)
    cb.set_label(r'$\langle v_z^2 \rangle$ [km/s]')

    ax = plt.subplot(gs[1, 3])
    img = ax.pcolor(xx, yy, VV[3], vmin=vminr, vmax=vmaxr, rasterized=ras)
    ax.set_ylabel('y [pc]')
    ax.set_xlabel('x [pc]')
    cb = plt.colorbar(img)
    cb.set_label(r'$\langle v_{\parallel}^2 \rangle$ [km/s]')

    ax = plt.subplot(gs[1, 4])
    img = ax.pcolor(xx, yy, VV[4], vmin=vminr, vmax=vmaxr, rasterized=ras)
    ax.set_ylabel('y [pc]')
    ax.set_xlabel('x [pc]')
    cb = plt.colorbar(img)
    cb.set_label(r'$\langle v_{\bot}^2 \rangle$ [km/s]')
    plt.tight_layout()
    plt.show()
