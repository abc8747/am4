# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, "..")
from plots import MPL

MPL.init()

base_path = Path("../../../../src/am4/utils/data/")
aircrafts = pq.read_table(base_path / "aircrafts.parquet")
aircrafts = aircrafts.filter(pc.field("type") != 1)
aircrafts
# %%
plt.switch_backend("webagg")
# %%
xs = np.array(aircrafts["cost"])
ys = np.array(aircrafts["speed"]) * np.array(aircrafts["capacity"])
zs = np.array(aircrafts["range"]) * 2 / np.array(aircrafts["speed"])
plt.close()
plt.scatter(xs, ys, c=zs, cmap="turbo", s=3, vmax=24)
for i, txt in enumerate(aircrafts["shortname"]):
    plt.text(xs[i], ys[i], txt, fontsize=6)
plt.xscale("log")
plt.yscale("log")
plt.colorbar()
plt.xlabel("Cost")
plt.ylabel("Speed")
plt.savefig("../../img/aircraft-research/.all.svg")
plt.show()


# %%
