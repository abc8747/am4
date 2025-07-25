# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

base_path = Path("../../../../src/am4/utils/data/")
airports = pq.read_table(base_path / "airports.parquet")
routes = pq.read_table(base_path / "routes.parquet")

yds = np.array(routes["yd"], dtype=np.uint16)
jds = np.array(routes["jd"], dtype=np.uint16)
fds = np.array(routes["fd"], dtype=np.uint16)
hcs = np.array(airports["hub_cost"])
lats = np.array(airports["lat"])
lngs = np.array(airports["lng"])
rwy = np.array(airports["rwy"])
hcs = np.array(airports["hub_cost"])
iata = np.array(airports["iata"])
icao = np.array(airports["icao"])

# %%
yds_matrix = np.empty((3907, 3907), dtype=np.float32)
yds_matrix.fill(np.nan)
jds_matrix = np.empty((3907, 3907), dtype=np.float32)
jds_matrix.fill(np.nan)
fds_matrix = np.empty((3907, 3907), dtype=np.float32)
fds_matrix.fill(np.nan)

x, y = 0, 0
for vy, vj, vf in zip(yds, jds, fds):
    y += 1
    if y == 3907:
        x += 1
        y = x + 1
    yds_matrix[x, y] = vy
    yds_matrix[y, x] = vy
    jds_matrix[x, y] = vj
    jds_matrix[y, x] = vj
    fds_matrix[x, y] = vf
    fds_matrix[y, x] = vf

# %%
import sys

sys.path.insert(0, "..")
from plots import MPL

MPL.init()
# %%
# full D[i, j] matrix plots
import matplotlib.gridspec as gridspec

for matrix, name in zip([yds_matrix, jds_matrix, fds_matrix], ["yds", "jds", "fds"]):
    fig = plt.figure(figsize=(10, 8.684))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 10])
    gs.update(hspace=0.0, wspace=0.0)

    ax1 = plt.subplot(gs[0])
    ax1.set_aspect("auto")
    ax1.tick_params(labelbottom=False, labelleft=False)
    cax1 = ax1.imshow(hcs.reshape(1, -1), interpolation="none", cmap="turbo", aspect="auto")

    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.set_aspect(1)
    cax2 = ax2.imshow(matrix, interpolation="none", cmap="turbo", aspect="auto")

    fig.colorbar(cax1, ax=ax1, orientation="vertical", label="Hub Cost", shrink=0.8)
    fig.colorbar(cax2, ax=ax2, orientation="vertical", label=name, shrink=0.8)

    plt.tight_layout()
    plt.savefig(f"../../img/demand-research/matrix_{name}.svg")
    plt.savefig(f"../../img/demand-research/matrix_{name}.webp", dpi=300)
    plt.close()

# %%
# zoom into the matrix
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

x0, x1 = 3400, 3500
y0, y1 = 400, 500

fig, axs = plt.subplots(1, 3, figsize=(12, 5), layout="tight", sharey=True)

for ax, matrix, name in zip(axs, [yds_matrix, jds_matrix, fds_matrix], ["yds", "jds", "fds"]):
    im = ax.imshow(matrix[x0:x1, y0:y1], interpolation="none", cmap="turbo", extent=[y0, y1, x1, x0])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    fig.colorbar(im, cax=cax, orientation="horizontal")
    ax.set_title(name)

plt.tight_layout()
plt.savefig("../../img/demand-research/matrices_handzoom.svg")
plt.savefig("../../img/demand-research/matrices_handzoom.webp", dpi=300)
plt.close()
# %%
rows = 3907 * 3906 // 2
data = np.empty((rows, 3), dtype=np.uint32)

x, y = 0, 0
for i, (vy, vj, vf) in enumerate(zip(yds, jds, fds)):
    y += 1
    if y == 3907:
        x += 1
        y = x + 1
    hclow, hchigh = hcs[x], hcs[y]
    if hclow > hchigh:
        hclow, hchigh = hchigh, hclow
    data[i] = hclow, hchigh, vy

# %%
fig, ax = plt.subplots(figsize=(12, 6), layout="tight")
sc = ax.scatter(data[:, 0] + data[:, 1], data[:, 2], s=1, alpha=0.01, c=data[:, 0], cmap="turbo")
ax.set_xlabel("Sum of hub costs")
ax.set_ylabel("Y demand")
cb = plt.colorbar(sc, ax=ax, label="Minimum of hub costs")
cb.solids.set(alpha=1)
plt.savefig("../../img/demand-research/scatter_hubcost_demand.webp", dpi=300)
plt.close()
# %%
fig, ax = plt.subplots(figsize=(12, 5))

for matrix, name, color in zip([yds_matrix, jds_matrix, fds_matrix], ["yds", "jds", "fds"], ["b", "g", "r"]):
    ax.hist(matrix[x0:x1, y0:y1].flatten(), bins=np.arange(0, 600), alpha=0.5, label=name, color=color)

ax.set_title(f"airport id {y0}:{y1} {x0}:{x1}")
ax.set_xlabel("Demand")
ax.set_ylabel("Frequency")
ax.legend()

plt.tight_layout()
plt.savefig("../../img/demand-research/histograms_handfilter_hc48000.svg")
plt.close()

# %%

dsy, dsj, dsf = [], [], []
x, y = 0, 0
for vy, vj, vf in zip(yds, jds, fds):
    y += 1
    if y == 3907:
        x += 1
        y = x + 1
    if hcs[x] == 48000 and hcs[y] == 48000:
        dsy.append(vy)
        dsj.append(vj)
        dsf.append(vf)
len(dsy), len(dsj), len(dsf)
# %%

fig, ax = plt.subplots(figsize=(12, 5))
for matrix, name, color in zip([dsy, dsj, dsf], ["yds", "jds", "fds"], ["b", "g", "r"]):
    ax.hist(matrix, bins=np.arange(0, 600), alpha=0.5, label=name, color=color)

ax.set_title(f"hc0 == hc1 == 48000, N={len(dsy)}")
ax.set_xlabel("Demand")
ax.set_ylabel("Frequency")
ax.legend()

plt.tight_layout()
plt.savefig("../../img/demand-research/histograms_all_hc48000.svg")

# %%

# hc -> [demand]
ysmap = {}
jsmap = {}
fsmap = {}
x, y = 0, 0
for vy, vj, vf in zip(yds, jds, fds):
    y += 1
    if y == 3907:
        x += 1
        y = x + 1
    if (h := hcs[x]) == hcs[y]:
        if h not in ysmap:
            ysmap[h] = []
            jsmap[h] = []
            fsmap[h] = []
        ysmap[h].append(vy)
        jsmap[h].append(vj)
        fsmap[h].append(vf)

ysmap = dict(sorted(ysmap.items()))
jsmap = dict(sorted(jsmap.items()))
fsmap = dict(sorted(fsmap.items()))
[f"{k}: {len(v)}" for k, v in ysmap.items()]
# %%
fig, ax = plt.subplots(figsize=(12, 6))
ax.violinplot(ysmap.values(), showmeans=True)
ax.violinplot(jsmap.values(), showmeans=True)
ax.violinplot(fsmap.values(), showmeans=True)

ax.set_xticks(np.arange(1, len(ysmap.keys()) + 1))
ax.set_xticklabels([f"{k}\nN={len(v)}" for k, v in ysmap.items()])
ax.set_xlabel("hub_cost")
ax.set_ylabel("demand")
fig.suptitle("demand distribution for equal hub costs")

plt.savefig("../../img/demand-research/demand_equal_hubcosts.svg")

# %%
# prepare dataset

X = []
Y = []

x, y = 0, 0
for vy, vj, vf in zip(yds, jds, fds):
    y += 1
    if y == 3907:
        x += 1
        y = x + 1
    if hcs[x] == 48000 and hcs[y] == 48000:
        X.append((x, y, lats[x], lngs[x], rwy[x], lats[y], lngs[y], rwy[y]))
        Y.append((vy, vj, vf, vy + 2 * vj + 3 * vf))

xs, ys = zip(*sorted(zip(X, Y), key=lambda x: x[1][0], reverse=True))
# %%
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

pminj = [[432, 45, 26], [204, 45, 102]]
pmaxj = [[204, 180, 12], [96, 180, 48]]

plt.close()
fig = plt.figure(layout="tight")
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(
    [y[0] for y in ys], [y[1] for y in ys], [y[2] for y in ys], c=[y[3] for y in ys], cmap="turbo_r", alpha=0.1, s=0.05
)
ax.set_xlabel("y")
ax.set_ylabel("j")
ax.set_zlabel("f")
ax.view_init(elev=30, azim=-30)
cb = plt.colorbar(sc)
cb.set_label("y+2j+3f")
cb.solids.set(alpha=1)
k = 1.45
for y, j, f in pminj:
    ax.plot([0, y * k], [0, j * k], [0, f * k])
for y, j, f in pmaxj:
    ax.plot([0, y * k], [0, j * k], [0, f * k])
plt.savefig("../../img/demand-research/3d_y_2j_3f.webp", dpi=600)
# %%


def init():
    return (sc,)


def update(num):
    print(num)
    ax.view_init(azim=num)
    return (sc,)


ani = FuncAnimation(fig, update, frames=range(0, 360, 5), interval=1000 / 60)
ani.save("../../video/animation_y_2j_3f.mp4", writer="ffmpeg", fps=30, dpi=600)
# %%
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("WebAgg")

y2j3f = [y[3] for y in ys]

plt.close()
fig, ax = plt.subplots(figsize=(12, 6), layout="tight")
ax.hist(y2j3f, bins=list(range(min(y2j3f), max(y2j3f) + 1)))
ax.set_xlabel("y+2j+3f")
ax.set_ylabel("frequency")
plt.savefig("../../img/demand-research/histogram_y_2j_3f.svg")
plt.show()
# %%
mat = np.empty((3907, 3907), dtype=np.float32)
mat.fill(np.nan)
for x, y, z in zip([x[0] for x in xs], [x[1] for x in xs], y2j3f):
    mat[x][y] = z
    mat[y][x] = z
# %%

lins = [(x[0] * x[1]) for x in xs]
plt.close()
plt.hist(lins, bins=np.arange(0, np.max(lins)))
# plt.imshow(mat, cmap="turbo", interpolation="none")
# plt.imshow(mat, interpolation="none")
# plt.colorbar()
plt.show()
plt.savefig("../../img/demand-research/temp.svg")

# %%
# (172, 212, 79) 11
# (191, 116, 69) 10
# (220, 241, 53) 10
# (388, 126, 58) 10
# (142, 199, 71) 10
# (175, 146, 61) 10
# (268, 135, 21) 10
# (472, 109, 50) 9
# (303, 76, 70) 9
# (228, 154, 36) 9

points = {}
x, y = 0, 0
for vy, vj, vf in zip(yds, jds, fds):
    y += 1
    if y == 3907:
        x += 1
        y = x + 1
    ya = vy + 2 * vj + 3 * vf
    if hcs[x] == 48000 and hcs[y] == 48000:
        if ya not in points:
            points[ya] = []
        points[ya].append((vy, vj, vf))
        # if ya != 800:
        #     continue
        # points.append((vy, vj, vf))
        # data.append((x, y, lats[x], lngs[x], rwy[x], lats[y], lngs[y], rwy[y], iata[x], icao[x], iata[y], icao[y]))
# %%
import matplotlib.animation as animation
import matplotlib.ticker as ticker

ybds = (100, 700)
jbds = (25, 275)
fbds = (0, 150)

plt.close()
fig = plt.figure(figsize=(15, 5), layout="tight")
gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])
ax = fig.add_subplot(gs[0, 0], projection="3d")
ax.view_init(elev=30, azim=30)
y2j3f = 600
p = np.array(points[y2j3f])
ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=2)
# sc = ax.scatter([], [], [], s=2)
ax.set_xlabel("y")
ax.set_ylabel("j")
ax.set_zlabel("f")
ax.set_xlim(*ybds)
ax.set_ylim(*jbds)
ax.set_zlim(*fbds)
ax.set_title(f"y+2j+3f = {y2j3f}")
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(p[:, 1], bins=np.arange(*jbds))
# hist = ax2.hist([], bins=jbds)
ax2.set_xlim(*jbds)
ax2.set_xlabel("j")
ax2.set_ylabel("frequency")
ax2.set_ylim(0, 100)
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))

pminj = np.array(sorted(((y, j, f) for y, j, f in points[y2j3f] if j == 45), key=lambda x: x[0], reverse=True))
ax.plot(pminj[0, 0], pminj[0, 1], pminj[0, 2], "ro")
ax.plot(pminj[-1, 0], pminj[-1, 1], pminj[-1, 2], "ro")
pmaxj = np.array(sorted(((y, j, f) for y, j, f in points[y2j3f] if j == 180), key=lambda x: x[0], reverse=True))
ax.plot(pmaxj[0, 0], pmaxj[0, 1], pmaxj[0, 2], "go")
ax.plot(pmaxj[-1, 0], pmaxj[-1, 1], pmaxj[-1, 2], "go")
print(f"pminj: {pminj[0]}, {pminj[-1]}")
print(f"pmaxj: {pmaxj[0]}, {pmaxj[-1]}")
plt.show()

# def update(frame):
#     print(f"update {frame}")
#     p = np.array(points[frame])
#     sc._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
#     ax.set_title(f"y+2j+3f = {frame}")

#     ax2.cla()
#     hist = ax2.hist(p[:, 1], bins=np.arange(*jbds))
#     return sc, ax2

# ani = animation.FuncAnimation(fig, update, frames=sorted(points.keys()), interval=1000/60, blit=True)
# ani.save("assets/animation_y_2j_3f_2.mp4", writer="ffmpeg", fps=30, dpi=600)

# %%
import pandas as pd

arr = np.genfromtxt(".rand.csv", delimiter=",")
plt.matshow(arr)
plt.show()
# %%
