# %%
import pandas as pd

df = pd.read_csv("data.csv")
df["rownum"] = range(1, len(df) + 1)
df
# %%
dfu = df[df["state"] == "unlocked"]
is_updated = dfu["date"] > 20220130
xs = dfu["pax_ac"] + dfu["slots"]
ys = dfu["10ac"]

pd.DataFrame({"fleet_lim": xs[is_updated], "C10": ys[is_updated]}).to_clipboard(index=False)
# %%
from scipy.optimize import curve_fit

xsf, ysf = xs[is_updated], ys[is_updated]


def f(x, a, b):
    return a * b**x


popt, pcov = curve_fit(f, xsf, ysf)
a, b = popt

resids = ysf - f(xsf, a, b)
r2 = 1 - sum(resids**2) / sum((ysf - ysf.mean()) ** 2)

a, b, r2
# %%
import sys

sys.path.insert(0, "..")
from plots import MPL

MPL.init()

# %%
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
ax: plt.Axes
ax.set_yscale("log")
ax.set_xlabel("fleet limit (slots + pax_ac)")
ax.set_ylabel("$C_{10}$")
ax.scatter(xs, ys, s=5)
for x, y, txt, date in zip(xs, ys, dfu["rownum"], is_updated):
    ax.text(x, y, f"{txt}{'*' if not date else ''}", fontsize=8, ha="center")

plt.savefig("../../img/hangar-10.svg")
plt.show()

# %%
