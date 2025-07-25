from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm


class MPL:
    @staticmethod
    def init():
        font_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "src"
            / "am4"
            / "bot"
            / "assets"
            / "font"
            / "B612-Regular.ttf"
        )
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)

        # friendly to dark and light mode
        matplotlib.rcParams.update(
            {
                "font.family": prop.get_name(),
                "figure.facecolor": "none",
                "axes.facecolor": "none",
                "axes.edgecolor": "#888",
                "axes.labelcolor": "#888",
                "text.color": "#888",
                "xtick.color": "#888",
                "ytick.color": "#888",
            }
        )
