#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as p
from scipy.stats import binned_statistic_2d
from rotation_model2 import Rotation
from astropy.coordinates import SkyCoord
from astropy import coordinates as C
from astropy import coordinates as coord
from astropy import units as u
from numpy.random import rand, randn
from corner import corner
from sys import argv
from matplotlib.patches import Circle

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from gala.units import galactic

import matplotlib as mpl
import matplotlib.cm as cm

import matplotlib
if (matplotlib.__version__ < '1.2'):
    from matplotlib.nxutils import points_inside_poly
else:
    from matplotlib.path import Path as mpl_path

def drawArrow(A, B):
    p.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
              head_width=10, length_includes_head=True, color='r')

def inside_poly(data, vertices):
    if (matplotlib.__version__ < '1.2'):
        return points_inside_poly(data, vertices)
    return mpl_path(vertices).contains_points(data)

def logistic(x, k, b):
    return 1./(1+np.exp(-k*(x-b)))

def ilogistic(x, k, b):
    return 1./(1+np.exp(-k*(-x+b)))

def getLvec(coo, pmra, pmdec, Vlos, rt):
    RA = coo.ra
    DEC = coo.dec
    Rsun = coo.distance
    pmRA = pmra*u.mas/u.yr
    pmDEC = pmdec*u.mas/u.yr
    Vlos = Vlos*u.km/u.s

    X = coo.galactocentric.x.value
    Y = coo.galactocentric.y.value
    Z = coo.galactocentric.z.value

    rsun = 8.3*u.kpc
    vsun = 239*u.km/u.s
    #Sun's velocity in astropy representation
    v_sun = C.CartesianDifferential([11.1, 239+12.24, 7.25])*u.km/u.s
    # this is v_pec + vsun; This is Schoenrich + (0, 239, 0)

    coo = C.ICRS(ra=RA, dec=DEC, distance=Rsun, pm_ra_cosdec=pmRA, pm_dec=pmDEC,
                 radial_velocity=Vlos)
    gc = coo.transform_to(C.Galactocentric(galcen_distance=rsun,
                                           galcen_v_sun=v_sun, z_sun=27*u.pc))

    x, y, z, vx, vy, vz = (gc.x, gc.y, gc.z, gc.v_x, gc.v_y, gc.v_z)

    R = np.array([x.value,y.value,z.value])
    VV = np.array([vx.value,vy.value,vz.value])

    L = np.cross(R,VV)
    L = L/np.linalg.norm(L)

    ## 100 pc offset
    rt = coo.distance.kpc*np.tan(np.deg2rad(rt/60.))
    x2 = X + 0.01*L[0]
    y2 = Y + 0.01*L[1]
    z2 = Z + 0.01*L[2]

    x3 = X - rt*L[0]
    y3 = Y - rt*L[1]
    z3 = Z - rt*L[2]

    coo2 = C.Galactocentric(x=[x2,x3]*u.kpc, y=[y2,y3]*u.kpc, z=[z2,z3]*u.kpc,
                            galcen_distance=8.3*u.kpc, z_sun=27*u.pc)

    t2 = coo2.transform_to(C.ICRS())
    return t2


def ortoproj(x, y, vx, vy, xc, yc):
    """ Compute X and Y in radians in the ortographic projection
    The minus sign in X is to align it with RA direction """
    D2R = np.pi/180.
    XX = -np.cos(D2R*y)*np.sin(D2R*(x - xc))
    YY = (np.sin(D2R*y)*np.cos(D2R*yc) -
         np.cos(D2R*y)*np.sin(D2R*yc)*np.cos(D2R*(x - xc)))

    oVX = vx*np.cos(D2R*(x-xc)) - vy*np.sin(D2R*y)*np.sin(D2R*(x-xc))
    oVY = vx*np.sin(D2R*xc)*np.sin(D2R*(x-xc)) + vy*(np.cos(D2R*y)*np.cos(D2R*yc) +   \
          np.sin(D2R*y)*np.sin(D2R*yc)*np.cos(D2R*(x-xc)))

    return(XX, YY, oVX, oVY)

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

## NGC 3201
K = 4.74047
vx0 = 8.35
vy0 = -2.00
vr0 = 494.34
d = 4.51
aa = SkyCoord("10h17m36.82s -46d24m44.9s", distance=d*u.kpc)
xc = aa.ra.deg
yc = aa.dec.deg

# Get orbit
vxvv = [xc, yc, d, vx0, vy0, vr0]
o = Orbit(vxvv=vxvv, radec=True, ro=8.1*u.kpc, vo=223*u.km/u.s)
t = np.linspace(0,0.01, 10)
o.integrate(t, MWPotential2014)
fo = o.flip()
fo.integrate(t, MWPotential2014)
ora = np.r_[fo.ra(t)[::-1], o.ra(t)]
odec = np.r_[fo.dec(t)[::-1], o.dec(t)]

## Mock stream
pal5_c = coord.SkyCoord(ra=xc*u.degree, dec=yc*u.degree,
                        distance=d*u.kpc,
                        pm_ra_cosdec=vx0*u.mas/u.yr,
                        pm_dec=vy0*u.mas/u.yr,
                        radial_velocity=vr0*u.km/u.s)
rep = pal5_c.transform_to(coord.Galactocentric).data
pal5_w0 = gd.PhaseSpacePosition(rep)
pal5_mass = 2.5e5 * u.Msun
pal5_pot = gp.PlummerPotential(m=pal5_mass, b=4*u.pc, units=galactic)
mw = gp.BovyMWPotential2014()
pal5_orbit = gp.Hamiltonian(mw).integrate_orbit(pal5_w0, dt=-0.001*u.Myr, n_steps=20000)

stream = ms.fardal_stream(mw, pal5_orbit[::-1], prog_mass=1E4*u.Msun, release_every=1)
#ms.fardal_stream
#fig = pal5_orbit.plot()
#fig = stream.plot(marker='.', s=1, alpha=0.25)
stream_c = stream.to_coord_frame(coord.ICRS)

Sra = stream_c.ra.deg
Sdec = stream_c.dec.deg
SRv = stream_c.radial_velocity.value
Spmra = stream_c.pm_ra_cosdec.value
Spmdec = stream_c.pm_dec.value
SXrad, SYrad, SoVX, SoVY = ortoproj(Sra, Sdec, Spmra, Spmdec, xc, yc)

## X and Y in pc
SX = d*1000*SXrad
SY = d*1000*SYrad


#p.plot(SX, SY, 'o')
#p.show()
#exit()

## X and Y in arcmin
SXp = (10800/np.pi)*SXrad
SYp = (10800/np.pi)*SYrad
SVX = K*d*(SoVX - vx0)
SVY = K*d*(SoVY - vy0)
SVZ = SRv - vr0



oX, oY, _, _ = ortoproj(ora, odec, 0, 0, xc, yc)
oX *= d*1000
oY *= d*1000
kk = (np.abs(oX)<120)*(np.abs(oY)<120)
oX = oX[kk]
oY = oY[kk]

# Get orbital L vector
LL = getLvec(aa, vx0, vy0, vr0, 15)
LLra = LL.ra.deg
LLdec = LL.dec.deg
LLX, LLY, _, _ = ortoproj(LLra, LLdec, 0, 0, xc, yc)
LLX *= d*1000
LLY *= d*1000

#p.plot(ora, odec)
#p.plot(LL.ra.deg, LL.dec.deg, 'r-', lw=3)
#p.show()
#exit()

x, y, vx, vy, vr, evx, evy, evr = np.loadtxt('matched.dat', unpack=True)
Xrad, Yrad, oVX, oVY = ortoproj(x, y, vx, vy, xc, yc)

vr0 = np.mean(vr)

## X and Y in pc
X = d*1000*Xrad
Y = d*1000*Yrad

## X and Y in arcmin
Xp = (10800/np.pi)*Xrad
Yp = (10800/np.pi)*Yrad

vxpersp = -6.1363e-5*Xp*vr0/d
vypersp = -6.1363e-5*Yp*vr0/d
vrpersp = 1.3790e-3*(Xp*vx0 + Yp*vy0)*d

VX = K*d*(oVX - vx0 - vxpersp)
VY = K*d*(oVY - vy0 - vypersp)
VZ = vr - vr0 - vrpersp
EVX = K*d*evx
EVY = K*d*evy
EVZ = evr
R = np.sqrt(X*X + Y*Y)

## Scipy voronoi does not bin, but assume each source is the centre of a cell
#jj = np.random.rand(len(X)) < 0.5
#vor = Voronoi(np.c_[X[jj], Y[jj]], incremental=True)
## find min/max values for normalization
#minima = min(VZ)
#maxima = max(VZ)
#
## normalize chosen colormap
#norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
#mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
#
## plot Voronoi diagram, and fill finite regions with color mapped from speed value
#voronoi_plot_2d(vor, show_points=False, show_vertices=False, s=1, line_alpha=0)
#for r in range(len(vor.point_region)):
#    region = vor.regions[vor.point_region[r]]
#    if not -1 in region:
#        polygon = [vor.vertices[i] for i in region]
#        inside = inside_poly(np.c_[X, Y], polygon)
#        print(len(VZ[inside]))
#        p.fill(*zip(*polygon), color=mapper.to_rgba(np.mean(VZ[inside])))

ip    = 90
PAp   = 45
wp    = 0.0
Rp    = 7
isol  = 30
PAsol = -160
wsol  = 0.1
K     = 0.1
Rb = 83.46/2.
Rs = 2

bins = 8

Hox, xe, ye, bn = binned_statistic_2d(X, Y, VX, statistic=np.nanmean, bins=bins, range=[[-80, 80], [-80, 80]])
Hoy, xe, ye, bn = binned_statistic_2d(X, Y, VY, statistic=np.nanmean, bins=bins, range=[[-80, 80], [-80, 80]])
Hoz, xe, ye, bn = binned_statistic_2d(X, Y, VZ, statistic=np.nanmean, bins=bins, range=[[-80, 80], [-80, 80]])

SHox, xe, ye, bn = binned_statistic_2d(SX, SY, SVX, statistic=np.nanmean, bins=bins, range=[[-80, 80], [-80, 80]])
SHoy, xe, ye, bn = binned_statistic_2d(SX, SY, SVY, statistic=np.nanmean, bins=bins, range=[[-80, 80], [-80, 80]])
SHoz, xe, ye, bn = binned_statistic_2d(SX, SY, SVZ, statistic=np.nanmean, bins=bins, range=[[-80, 80], [-80, 80]])


Hoxx, xe, ye, bn = binned_statistic_2d(X, Y, VX, statistic=np.nanstd, bins=bins, range=[[-80, 80], [-80, 80]])
Hoyy, xe, ye, bn = binned_statistic_2d(X, Y, VY, statistic=np.nanstd, bins=bins, range=[[-80, 80], [-80, 80]])
Hozz, xe, ye, bn = binned_statistic_2d(X, Y, VZ, statistic=np.nanstd, bins=bins, range=[[-80, 80], [-80, 80]])

## Observed mean and dispersion (simple, not P&M)
Hox = Hox.T
Hoy = Hoy.T
Hoz = Hoz.T
Hoxx = Hoxx.T
Hoyy = Hoyy.T
Hozz = Hozz.T

Mp = Rotation(i=ip, PA=PAp, kind='rpeak', w=wp, Rpeak=Rp)
Ms = Rotation(i=isol, PA=PAsol, kind='solid', w=wsol)

x, y = np.arange(-80, 80, 3), np.arange(-80, 80, 3)
xx, yy = np.meshgrid(x, y)
#xxp, yyp = xx*10800/(1000*d*np.pi), yy*10800/(1000*d*np.pi)

xxp = xx/(d*1000) # radians
xxp = np.rad2deg(xxp)*60 # arcmin
yyp = yy/(d*1000) # radians
yyp = np.rad2deg(yyp)*60 # arcmin

VperspX = -(K*d)*6.1363e-5*xxp*vr0/d
VperspY = -(K*d)*6.1363e-5*yyp*vr0/d
VperspZ =  1.3790e-3*(xxp*vx0 + yyp*vy0)*d

rr = np.sqrt(xx*xx + yy*yy)

Vs = Ms.m1v(xx, yy, Rs)
Vp = Mp.m1v(xx, yy, Rs)

VVs = Ms.m2v(xx, yy, Rs)
VVp = Mp.m2v(xx, yy, Rs)

V = logistic(rr, K, Rb)*Vs + ilogistic(rr, K, Rb)*Vp
VV = logistic(rr, K, Rb)*VVs + ilogistic(rr, K, Rb)*VVp ## <v^2> = Var = Disp^2
V = Vs
VV = VVs

## Variance (from fit to data) over expectation from rotation alone
#VVsig = VV
#VVsig[0] = np.sqrt(VV[0]/var[0])
#VVsig[1] = np.sqrt(VV[1]/var[1])
#VVsig[2] = np.sqrt(VV[2]/var[2])

### Figure 2 ###

import matplotlib.gridspec as gridspec
p.figure(figsize=(12, 8))
gs = gridspec.GridSpec(4, 3)

ras = True
vmin = None
vmax = None
vminr = -3
vmaxr = 3

vminp =  None
vmaxp =  None
vminrp = None
vmaxrp = None

ax = p.subplot(gs[0, 0])
img = ax.pcolor(xx, yy, V[0], vmin=vmin, vmax=vmax, rasterized=ras)
cb = p.colorbar(img)
cb.set_label('vx')
cb.set_label(r'$\langle v_x \rangle$ [km/s]')
p.plot(oX, oY, 'k-', lw=3, alpha=0.5)
drawArrow(LLX, LLY)
p.xlim(-80, 80)


ax = p.subplot(gs[0, 1])
img = ax.pcolor(xx, yy, V[1], vmin=vmin, vmax=vmax, rasterized=ras)
cb = p.colorbar(img)
cb.set_label(r'$\langle v_y \rangle$ [km/s]')
p.plot(oX, oY, 'k-', lw=3, alpha=0.5)
drawArrow(LLX, LLY)
p.xlim(-80, 80)

ax = p.subplot(gs[0, 2])
img = ax.pcolor(xx, yy, V[2], vmin=vminr, vmax=vmaxr, rasterized=ras)
cb = p.colorbar(img)
cb.set_label(r'$\langle v_z \rangle$ [km/s]')
p.plot(oX, oY, 'k-', lw=3, alpha=0.5)
drawArrow(LLX, LLY)
p.xlim(-80, 80)

ax = p.subplot(gs[1, 0])
img = ax.pcolor(xe, ye, Hox, vmin=vmin, vmax=vmax, rasterized=ras)
ax.set_ylabel('y [pc]')
cb = p.colorbar(img)
cb.set_label(r'$\langle v_x \rangle_{obs}$ [km/s]')
p.plot(oX, oY, 'k-', lw=3, alpha=0.5)
drawArrow(LLX, LLY)
#p.plot(X, Y, 'k.')

ax = p.subplot(gs[1, 1])
img = ax.pcolor(xe, ye, Hoy, vmin=vmin, vmax=vmax, rasterized=ras)
cb = p.colorbar(img)
cb.set_label(r'$\langle v_y \rangle_{obs}$ [km/s]')
p.plot(oX, oY, 'k-', lw=3, alpha=0.5)
drawArrow(LLX, LLY)

ax = p.subplot(gs[1, 2])
img = ax.pcolor(xe, ye, Hoz, vmin=vminr, vmax=vmaxr, rasterized=ras)
cb = p.colorbar(img)
cb.set_label(r'$\langle v_z \rangle_{obs}$ [km/s]')
patch = Circle((0,0), radius=Rb, fill=False)
ax.add_patch(patch)
p.plot(oX, oY, 'k-', lw=3, alpha=0.5)
drawArrow(LLX, LLY)

ax = p.subplot(gs[2, 0])
img = ax.pcolor(xx, yy, VperspX, vmin=vminp, vmax=vmaxp, rasterized=ras)
cb = p.colorbar(img)
cb.set_label(r'$v_{x, persp}$ [km/s]')

ax = p.subplot(gs[2, 1])
img = ax.pcolor(xx, yy, VperspY, vmin=vminp, vmax=vmaxp, rasterized=ras)
ax.set_xlabel('x [pc]')
cb = p.colorbar(img)
cb.set_label(r'$\langle v_y \rangle_{obs}$ [km/s]')
cb.set_label(r'$v_{y, persp}$ [km/s]')
#p.plot(X, Y, 'k.')

ax = p.subplot(gs[2, 2])
img = ax.pcolor(xx, yy, VperspZ, vmin=vminrp, vmax=vmaxrp, rasterized=ras)
cb = p.colorbar(img)
cb.set_label(r'$v_{z, persp}$ [km/s]')
#p.plot(X, Y, 'k.')

ax = p.subplot(gs[3, 0])
img = ax.pcolor(xe, ye, SHox, vmin=vmin, vmax=vmax, rasterized=ras)
cb = p.colorbar(img)
cb.set_label(r'$v_{x, stream}$ [km/s]')

ax = p.subplot(gs[3, 1])
img = ax.pcolor(xe, ye, SHoy, vmin=vmin, vmax=vmax, rasterized=ras)
ax.set_xlabel('x [pc]')
cb = p.colorbar(img)
cb.set_label(r'$\langle v_y \rangle_{obs}$ [km/s]')
cb.set_label(r'$v_{y, stream}$ [km/s]')
#p.plot(X, Y, 'k.')

ax = p.subplot(gs[3, 2])
img = ax.pcolor(xe, ye, SHoz, vmin=vminr, vmax=vmaxr, rasterized=ras)
cb = p.colorbar(img)
cb.set_label(r'$v_{z, stream}$ [km/s]')
#p.plot(X, Y, 'k.')

p.tight_layout()

if argv[1]=='pub':
    print('publishing figures')
    p.savefig(bdir+'databestfitvel.pdf')
    p.savefig(bdir+'databestfitvel.png')


p.show()

#p.figure(figsize=(12, 6))
#gs = gridspec.GridSpec(1, 3)
#
#ras = True
#
#ax = p.subplot(gs[0, 0])
#img = ax.pcolor(xx, yy, VVsig[0], rasterized=ras)
#ax.set_ylabel('y [pc]')
#ax.set_xlabel('x [pc]')
#cb = p.colorbar(img)
#cb.set_label('vx')
#cb.set_label(r'$Var/\langle v_x^2 \rangle$ [km/s]')
#
#ax = p.subplot(gs[0, 1])
#img = ax.pcolor(xx, yy, VVsig[1], rasterized=ras)
#ax.set_ylabel('y [pc]')
#ax.set_xlabel('x [pc]')
#cb = p.colorbar(img)
#cb.set_label(r'$Var/\langle v_y^2 \rangle$ [km/s]')
#
#ax = p.subplot(gs[0, 2])
#img = ax.pcolor(xx, yy, VVsig[2], rasterized=ras)
#ax.set_ylabel('y [pc]')
#ax.set_xlabel('x [pc]')
#cb = p.colorbar(img)
#cb.set_label(r'$Var/\langle v_z^2 \rangle$ [km/s]')
#
#p.tight_layout()
#p.show()
#exit()
