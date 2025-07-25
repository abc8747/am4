# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "..")
from plots import MPL

MPL.init()

base_path = Path("../../../../src/am4/utils/data/")
airports = pq.read_table(base_path / "airports.parquet")
airports
# %%
plt.switch_backend("webagg")
# %%
from math import asin, atan2, cos, degrees, radians, sin

import numpy as np


def get_circle_coordinates(lat, lng, radius, num_points=360):
    lat, lng = radians(lat), radians(lng)

    d_ang = radius / 6371
    bearings = np.linspace(0, 2 * np.pi, num_points)

    new_lat = np.arcsin(np.sin(lat) * np.cos(d_ang) + np.cos(lat) * np.sin(d_ang) * np.cos(bearings))
    new_lon = lng + np.arctan2(
        np.sin(bearings) * np.sin(d_ang) * np.cos(lat),
        np.cos(d_ang) - np.sin(lat) * np.sin(new_lat),
    )

    return np.array([np.degrees(new_lon), np.degrees(new_lat)])


# x, y
# %%
from scipy.stats import gaussian_kde

lng = airports["lng"].to_numpy()
lat = airports["lat"].to_numpy()
costs = airports["hub_cost"].to_numpy()

kde = gaussian_kde(np.vstack([lng, lat]), bw_method=0.03)

N = 300
xgrid = np.linspace(-180, 180, N * 2)
ygrid = np.linspace(-90, 90, N)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))


plt.close()
plt.figure(figsize=(16, 8))
plt.imshow(
    Z.reshape(Xgrid.shape),
    origin="lower",
    aspect="auto",
    extent=[-180, 180, -90, 90],
    cmap="turbo",
)
plt.scatter(lng, lat, alpha=costs / np.max(costs), s=1, c="white")
for d in range(1000, 15000, 1000):
    circ = get_circle_coordinates(40, 0, d)
    plt.plot(*circ, c="red")
plt.colorbar(label="density")
plt.show()

# %%
