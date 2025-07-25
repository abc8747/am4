# ruff: noqa
# fmt: off
#%%
# --8<-- [start:normal-eq]
import numpy as np

def lobf(xs: list[float], ys: list[float]) -> tuple[float, float]:
    A = np.array([np.ones_like(xs), xs]).T
    b = np.array(ys)
    return tuple(np.linalg.inv(A.T @ A) @ A.T @ b)
# --8<-- [end:normal-eq]
#%%
"""
# --8<-- [start:normal-eq-output]
>>> xs = [1,2,3]
>>> ys = [1,3,2]
>>> lobf(xs, ys)
(1, 0.5)
>>> np.polyfit(xs, ys, 1)
array([0.5, 1. ])
# --8<-- [end:normal-eq-output]
"""
#%%
from plots import MPL
MPL.init()

# --8<-- [start:cont-plot]
import matplotlib.pyplot as plt
import numpy as np

def ci(d, v, tp):
    ci_ok = (d >= 0.3*v*tp) & (d <= v*tp)
    ci = 2000*d / (7*v*tp) - 600 / 7
    return np.where(ci_ok, ci, np.nan)

def cont(d, v, tp):
    k = np.where(d < 6000, 0.0064, np.where(d < 10000, 0.0032, 0.0048))
    cont = np.minimum(k*d*(3-ci(d, v, tp)/100), 152)
    return cont/tp

dconst, vconst, tpconst = 6000-1e-9, 1049*1.1*1.5, 12
ds = np.linspace(0, 20000, 1000)
vs = np.linspace(300, 1800, 1000)
tps = np.linspace(0, 16, 1000)

fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(1, 3)

vm, tpm = np.meshgrid(vs, tps)
ax0 = fig.add_subplot(gs[0, 0])
ct0 = ax0.contourf(vm, tpm, cont(dconst, vm, tpm), levels=20, cmap='turbo_r')
cb0 = fig.colorbar(ct0, ax=ax0, location="bottom")

dm, tpm = np.meshgrid(ds, tps)
ax1 = fig.add_subplot(gs[0, 1])
ct1 = ax1.contourf(dm, tpm, cont(dm, vconst, tpm), levels=20, cmap='turbo_r')
cb1 = fig.colorbar(ct1, ax=ax1, location="bottom")

dm, vm = np.meshgrid(ds, vs)
ax2 = fig.add_subplot(gs[0, 2])
ct2 = ax2.contourf(dm, vm, cont(dm, vm, tpconst), levels=20, cmap='turbo_r')
cb2 = fig.colorbar(ct2, ax=ax2, location="bottom")
# --8<-- [end:cont-plot]

dlabel = 'Distance $d$, km'
vlabel = 'Original speed $v$, km/h'
tlabel = 'Time $T\'$, hr'
for ax, xlabel, ylabel, title, cb in zip(
    [ax0, ax1, ax2],
    [vlabel, dlabel, dlabel],
    [tlabel, tlabel, vlabel],
    [f'$d$ = {dconst:.0f} km', f'$v$ = {vconst:.2f} km/h (A388 with speed mod)', f'$T\'$ = {tpconst} hr'],
    [cb0, cb1, cb2]
):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cb.set_label('Contribution per hour, $\$_{\\text{C}}/T\'$')

plt.tight_layout()
plt.savefig('../img/contribution-per-hr-contour.svg')
plt.show()
plt.close(fig)
# %%
fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(1, 2)

# vconst = 1000
vm, tpm = np.meshgrid(vs, tps)
ax0 = fig.add_subplot(gs[0, 0])
ct0 = ax0.contourf(vm, tpm, cont(dconst, vm, tpm), levels=20, cmap='turbo_r')
ax0.axvline(vconst, color='#888', lw=2)
ax0.text(vconst-10, .1, f'{vconst:.2f} km/h', ha='right', va='bottom', color='#888')
ax0.set_xlabel(vlabel)
ax0.set_ylabel(tlabel)
ax0.set_title(f'$d$ = {dconst:.0f} km')
cb0 = fig.colorbar(ct0, ax=ax0, location="bottom")
cb0.set_label('Contribution per hour, $\$_{\\text{C}}/T\'$')

ax1 = fig.add_subplot(gs[0, 1])
ys = cont(dconst, vconst, tps)
ax1.plot(tps, ys, color='#888', lw=2)
tpopt = 40/27 * dconst / vconst
jopt = cont(dconst, vconst, tpopt)
ax1.axvline(tpopt, color='#548c38', lw=1)
ax1.scatter(tpopt, jopt, color='#548c38', s=10)
ax0.axhline(tpopt, color='#548c38', lw=.5)
print(tpopt, jopt)

ax1.set_xlim(0, 16)
ax1.set_ylim(bottom=0)
ax1.set_xlabel(tlabel)
ax1.set_ylabel('Contribution per hour, $\$_{\\text{C}}/T\'$')
ax1.set_title(f'$d$ = {dconst:.0f} km, $v$ = {vconst:.0f} km/h')

plt.tight_layout()
plt.savefig('../img/contribution-per-hr-speed-cut.svg')
plt.show()
plt.close(fig)
#%%
def tpopt(d, v):
    return 40/27 * d / v

fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(1, 2)

vm, tpm = np.meshgrid(vs, tps)
ax0 = fig.add_subplot(gs[0, 0])
ct0 = ax0.contourf(vm, tpm, cont(dconst, vm, tpm), levels=20, cmap='turbo_r')
ax0.plot(vs, tpopt(dconst, vs), color='#548c38', lw=1)
ax0.set_ylim(0, 16)
vssamples = np.arange(600, 1800, 200)
ax0.scatter(vssamples, tpopt(dconst, vssamples), color='#548c38', s=10)

ax0.set_xlabel(vlabel)
ax0.set_ylabel(tlabel)
ax0.set_title(f'$d$ = {dconst:.0f} km')
cb0 = fig.colorbar(ct0, ax=ax0, location="bottom")
cb0.set_label('Contribution per hour, $\$_{\\text{C}}/T\'$')

ax1 = fig.add_subplot(gs[0, 1])
for vsample in vssamples:
    ys = cont(dconst, vsample, tps)
    ax1.plot(tps, ys, color='#888', lw=.5)
    tsample = tpopt(dconst, vsample)
    jsample = cont(dconst, vsample, tsample)
    ax1.text(tsample, jsample, f'{vsample:.0f} km/h', ha='center', va='bottom', color='#555')
    ax1.scatter(tsample, jsample, color='#548c38', s=10)

ax1.set_xlim(0, 16)
ax1.set_ylim(bottom=0)
ax1.set_xlabel(tlabel)
ax1.set_ylabel('Contribution per hour, $\$_{\\text{C}}/T\'$')
ax1.set_title(f'$d$ = {dconst:.0f} km')

plt.tight_layout()
plt.savefig('../img/contribution-per-hr-speed-cut-multiple.svg')
plt.show()
plt.close(fig)

# %%
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 2)

dm, vm = np.meshgrid(ds, vs)
ax1 = fig.add_subplot(gs[0, 0])
ct1 = ax1.contourf(dm, vm, cont(dm, vm, 8), levels=20, cmap='turbo_r')

ax2 = fig.add_subplot(gs[0, 1])
ct2 = ax2.contourf(dm, vm, cont(dm, vm, 12), levels=20, cmap='turbo_r')

vmin = min(ct1.get_array().min(), ct2.get_array().min())
vmax = max(ct1.get_array().max(), ct2.get_array().max())

norm = colors.Normalize(vmin=vmin, vmax=vmax)
ct1.set_norm(norm)
ct2.set_norm(norm)

dlabel = 'Distance $d$, km'
vlabel = 'Original speed $v$, km/h'
for ax, xlabel, ylabel, title in zip(
    [ax1, ax2],
    [dlabel, dlabel],
    [vlabel, vlabel],
    [f'$T\'$ = 8 hr', f'$T\'$ = 12 hr']
):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    v = 1049*1.1
    ax.axhline(v*1.5, color='#888', lw=2)
    ax.text(20000, v*1.5, f'{v*1.5:.2f} km/h', ha='right', va='bottom', color='#888')
    ax.axhline(v, color='#888', lw=2)
    ax.text(20000, v, f'{v:.2f} km/h', ha='right', va='bottom', color='#888')

plt.tight_layout()
cb = fig.colorbar(ct1, ax=[ax1, ax2], location="right")
cb.set_label('Contribution per hour, $\$_{\\text{C}}/T\'$')
plt.savefig('../img/contribution-8h-12h-strat.svg')
plt.show()
plt.close(fig)
# %%
v, tp = 1049*1.1*1.5, 8

d = 27*v*tp/40
d, ci(d, v, tp), cont(d, v, tp) * 24

# %%
