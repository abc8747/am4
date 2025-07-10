from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import cmocean
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import PIL
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from pyproj import CRS, Transformer

from .utils import format_num

if TYPE_CHECKING:
    from .cogs.routes import HubProfitData


class MPLMap:
    def __init__(self):
        # setup the style and font
        font_path = Path(__file__).parent / "assets" / "font" / "B612-Regular.ttf"
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        self.rc_params = {
            "font.family": prop.get_name(),  # "B612"
            "axes.facecolor": "#16171a",
            "savefig.facecolor": "#1f2024",
            "legend.fontsize": 10 * 0.9,
            "legend.handlelength": 2 * 0.9,
            "text.color": "white",
            "axes.labelcolor": "white",
            "axes.edgecolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        }

        self.template_routes = self.create_routes_template()
        self.cmap = cmocean.tools.crop_by_percent(cmocean.cm.dense_r, 30, which="min")
        self.cmap2 = cmocean.tools.crop_by_percent(cmocean.cm.curl, 50)
        self.wgs84_to_pierceq = Transformer.from_crs(4326, CRS.from_string("+proj=peirce_q +lon_0=25 +shape=square"))

    def create_routes_template(self):
        """Create a blank template for the routes plot, which will be unpickled later to draw the plot."""
        plt.style.use("dark_background")
        ext = 2**24

        plt.rcParams.update(self.rc_params)

        fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), layout="tight")
        ax3 = ax2.twiny()

        ax: plt.Axes
        ax.set_axis_off()
        ax.set_xlim(-ext, ext)
        ax.set_ylim(-ext, ext)

        ax2: plt.Axes
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_xlabel("direct distance, km")
        ax2.set_ylabel("profit, $/d/ac")

        ax3: plt.Axes
        ax3.set_xlabel("#aircraft")
        ax3.invert_xaxis()
        d = Path(__file__).parent / "assets" / "img" / "map.jpg"  # peirce_quincuncial
        im = np.array(PIL.Image.open(d))

        # old bug eliminated: see https://github.com/matplotlib/matplotlib/issues/28448
        ext = 2**24
        ax.imshow(im.astype(np.uint16), extent=[-ext, ext, -ext, ext])

        template = pickle.dumps((fig, ax, ax2, ax3))
        plt.close(fig)
        return template

    def _plot_destinations(
        self,
        cols: dict[str, list],
        origin_lngs: list[float],
        origin_lats: list[float],
    ) -> io.BytesIO:
        fig, ax, ax2, ax3 = pickle.loads(self.template_routes)
        fig: Figure
        ax: plt.Axes
        ax2: plt.Axes
        ax3: plt.Axes

        lats = cols["98|dest.lat"]
        lngs = cols["99|dest.lng"]
        tpdpas = np.array(cols["32|trips_pd_pa"])
        profits = np.array(cols["39|profit_pt"]) * tpdpas
        sc_d = ax.scatter(*self.wgs84_to_pierceq.transform(lats, lngs), c=profits, s=0.5, cmap=self.cmap)
        ax.plot(*self.wgs84_to_pierceq.transform(origin_lats, origin_lngs), "ro", markersize=3)
        legend = ax.legend(*sc_d.legend_elements(fmt=FuncFormatter(format_num)), title="$/d/ac")

        ac_needs = cols["33|num_ac"]
        c = 0
        y1 = []
        for acn, pro in zip(ac_needs, profits):
            for _ in range(acn):
                y1.append(pro)
                c += 1

        binwidth = 10000
        bins = np.arange(min(y1), max(y1) + binwidth, binwidth)
        ax3.hist(y1, bins=bins, alpha=0.4, orientation="horizontal")

        dists = cols["30|direct_dist"]
        sc_tpdpa = ax2.scatter(dists, profits, s=1.5, c=tpdpas, cmap=self.cmap2)
        legend = ax2.legend(*sc_tpdpa.legend_elements(), title="t/d/ac", loc="upper left")
        ax2.add_artist(legend)

        buf = io.BytesIO()
        fig.savefig(buf, format="jpg", dpi=200)
        buf.seek(0)
        plt.close(fig)
        return buf

    def plot_hub_comparison(self, hubs_data: dict[str, HubProfitData]) -> io.BytesIO:
        plt.style.use("dark_background")
        plt.rcParams.update(self.rc_params)

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, width_ratios=[3, 1.2])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        ax_legend = fig.add_subplot(gs[:, 1])
        ax_legend.axis("off")

        def sort_key(item: tuple[str, HubProfitData]) -> float:
            profits = sorted(item[1].profits_per_ac, reverse=True)
            return sum(profits[:30])

        hubs_sorted = sorted(hubs_data.items(), key=sort_key, reverse=True)

        for iata, data in hubs_sorted:
            profits = np.array(sorted(data.profits_per_ac, reverse=True))
            if len(profits) == 0:
                continue

            hub_cost = data.hub_cost
            top10_profit = sum(profits[:10])
            top30_profit = sum(profits[:30])

            label = (
                f"{iata} | Top 10: ${format_num(top10_profit)}/d | "
                f"Top 30: ${format_num(top30_profit)}/d | Hub Cost: ${format_num(hub_cost)}"
            )

            num_aircraft = np.arange(1, len(profits) + 1)
            cum_profit = np.cumsum(profits)
            avg_top_k_profit = cum_profit / num_aircraft

            ax1.plot(num_aircraft, cum_profit, label=label, lw=1)
            ax2.plot(num_aircraft, profits, label=label, lw=1)
            ax3.plot(num_aircraft, avg_top_k_profit, label=label, lw=1)

        ax1.set_ylabel("cumulative Profit, $/d")
        ax1.yaxis.set_major_formatter(FuncFormatter(format_num))
        ax1.grid(True, linestyle="--", alpha=0.3)
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2.set_ylabel("profit per aircraft, $/d/ac")
        ax2.yaxis.set_major_formatter(FuncFormatter(format_num))
        ax2.grid(True, linestyle="--", alpha=0.3)
        plt.setp(ax2.get_xticklabels(), visible=False)

        ax3.set_xlabel("number of aircraft (sorted by profit, k)")
        ax3.set_ylabel("average top-k profit, $/d/ac")
        ax3.yaxis.set_major_formatter(FuncFormatter(format_num))
        ax3.grid(True, linestyle="--", alpha=0.3)

        handles, labels = ax1.get_legend_handles_labels()
        ax_legend.legend(handles, labels, loc="center left", fontsize="small", frameon=False)

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="jpg", dpi=200)
        buf.seek(0)
        plt.close(fig)
        return buf
