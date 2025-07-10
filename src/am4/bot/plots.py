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
import polars as pl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from pyproj import CRS, Transformer

from .utils import format_num

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .cogs.routes import HubProfitData


class MPLMap:
    def __init__(self):
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
        plt.style.use("dark_background")
        plt.rcParams.update(self.rc_params)

        self.template_routes = self.create_routes_template()
        self.template_hub_comparison = self.create_hub_comparison_template()

        self.dense_r = cmocean.tools.crop_by_percent(cmocean.cm.dense_r, 30, which="min")
        self.curl = cmocean.tools.crop_by_percent(cmocean.cm.curl, 50)
        self.wgs84_to_pierceq = Transformer.from_crs(4326, CRS.from_string("+proj=peirce_q +lon_0=25 +shape=square"))

    def create_routes_template(self):
        ext = 2**24
        fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), layout="tight")
        ax3 = ax2.twiny()

        ax: Axes
        ax.set_axis_off()
        ax.set_xlim(-ext, ext)
        ax.set_ylim(-ext, ext)

        ax2: Axes
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_xlabel("direct distance, km")
        ax2.set_ylabel("profit, $/d/ac")

        ax3: Axes
        ax3.set_xlabel("#aircraft")
        ax3.invert_xaxis()
        d = Path(__file__).parent / "assets" / "img" / "map.jpg"
        im = np.array(PIL.Image.open(d))

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
        ax: Axes
        ax2: Axes
        ax3: Axes

        lats = cols["98|dest.lat"]
        lngs = cols["99|dest.lng"]
        tpdpas = np.array(cols["32|trips_pd_pa"])
        profits = np.array(cols["39|profit_pt"]) * tpdpas
        sc_d = ax.scatter(*self.wgs84_to_pierceq.transform(lats, lngs), c=profits, s=0.5, cmap=self.dense_r)
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
        sc_tpdpa = ax2.scatter(dists, profits, s=1.5, c=tpdpas, cmap=self.curl)
        legend = ax2.legend(*sc_tpdpa.legend_elements(), title="t/d/ac", loc="upper left")
        ax2.add_artist(legend)

        buf = io.BytesIO()
        fig.savefig(buf, format="jpg", dpi=200)
        buf.seek(0)
        plt.close(fig)
        return buf

    def create_hub_comparison_template(self):
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, width_ratios=[1, 0.7])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        ax_legend = fig.add_subplot(gs[:, 1])
        ax_legend.axis("off")

        ax1.set_ylabel("profit per aircraft, $/d/ac")
        ax1.yaxis.set_major_formatter(FuncFormatter(format_num))
        ax1.grid(True, alpha=0.15)
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2.set_ylabel("cumulative profit, $/d")
        ax2.yaxis.set_major_formatter(FuncFormatter(format_num))
        ax2.grid(True, alpha=0.15)
        plt.setp(ax2.get_xticklabels(), visible=False)

        ax3.set_xlabel("number of aircraft (sorted by profit, k)")
        ax3.set_ylabel("average top-k profit, $/d/ac")
        ax3.yaxis.set_major_formatter(FuncFormatter(format_num))
        ax3.grid(True, alpha=0.15)

        fig.tight_layout()

        template = pickle.dumps((fig, ax1, ax2, ax3, ax_legend))
        plt.close(fig)
        return template

    def _plot_hub_comparison(self, hubs_data: dict[str, HubProfitData]) -> io.BytesIO:
        fig, ax1, ax2, ax3, ax_legend = pickle.loads(self.template_hub_comparison)
        fig: Figure
        ax1: Axes
        ax2: Axes
        ax3: Axes
        ax_legend: Axes

        hubs_metrics = []
        for iata, data in hubs_data.items():
            profits = np.array(sorted(data.profits_per_ac, reverse=True))
            if len(profits) == 0:
                continue
            hubs_metrics.append(
                {
                    "iata": iata,
                    "profits_raw": profits,
                    "top10": np.sum(profits[:10]),
                    "top30": np.sum(profits[:30]),
                    "top100": np.sum(profits[:100]),
                    "hub_cost": float(data.hub_cost),
                }
            )

        if not hubs_metrics:
            plt.close(fig)
            return io.BytesIO()

        df = pl.from_dicts(hubs_metrics).sort("top30", descending=True)

        cell_colors_data = {}

        for col_name in ["top10", "top30", "top100", "hub_cost"]:
            series = df[col_name]
            min_val, max_val = series.min(), series.max()
            if min_val is None or max_val is None or min_val == max_val:
                norm_values = np.full(len(series), 0.5)
            else:
                norm_values = (series - min_val) / (max_val - min_val)
            norm_values = ((1 - norm_values) if col_name == "hub_cost" else norm_values) * 0.4
            cell_colors_data[col_name] = [cmocean.cm.thermal(x * 0.3)[:3] for x in norm_values]

        table_data = []
        final_cell_colors = []
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        for i, hub in enumerate(df.to_dicts()):
            line_color = colors[i % len(colors)]
            table_data.append(
                [
                    hub["iata"],
                    f"${format_num(hub['top10'])}",
                    f"${format_num(hub['top30'])}",
                    f"${format_num(hub['top100'])}",
                    f"${format_num(hub['hub_cost'])}",
                ]
            )
            final_cell_colors.append(
                [
                    self.rc_params["axes.facecolor"],
                    cell_colors_data["top10"][i],
                    cell_colors_data["top30"][i],
                    cell_colors_data["top100"][i],
                    cell_colors_data["hub_cost"][i],
                ]
            )

            profits = hub["profits_raw"]
            num_aircraft = np.arange(1, len(profits) + 1)
            cum_profit = np.cumsum(profits)
            avg_top_k_profit = cum_profit / num_aircraft

            ax1.plot(num_aircraft, profits, color=line_color, lw=1)
            ax2.plot(num_aircraft, cum_profit, color=line_color, lw=1)
            ax3.plot(num_aircraft, avg_top_k_profit, color=line_color, lw=1)

        columns = ["Hub", "Top 10/d", "Top 30/d", "Top 100/d", "Hub cost"]
        header_colors = [self.rc_params["axes.facecolor"]] * len(columns)

        table = ax_legend.table(
            cellText=table_data,
            colLabels=columns,
            loc="center",
            cellLoc="center",
            colColours=header_colors,
            cellColours=final_cell_colors,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.8)

        for i, hub in enumerate(df.to_dicts()):
            table[i + 1, 0].get_text().set_color(colors[i % len(colors)])

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="jpg", dpi=200)
        buf.seek(0)
        plt.close(fig)
        return buf
