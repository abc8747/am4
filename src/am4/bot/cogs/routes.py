import asyncio
import io
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import discord
import orjson
import pyarrow as pa
from discord.ext import commands
from pyarrow import csv

from am4.utils.aircraft import Aircraft
from am4.utils.airport import Airport
from am4.utils.game import User
from am4.utils.route import AircraftRoute, Destination, RoutesSearch

from ...config import cfg
from ..base import BaseCog
from ..converters import AircraftCvtr, CfgAlgCvtr, ConstraintCvtr, MultiAirportCvtr, TPDCvtr
from ..errors import CustomErrHandler
from ..plots import MPLMap
from ..utils import (
    COLOUR_WARNING,
    HELP_CFG_ALG,
    HELP_TPD,
    ICSV,
    IJSON,
    fetch_user_info,
    format_ap_short,
    format_config,
    format_demand,
    format_flight_time,
    format_ticket,
    get_realism_departure_runway_warning,
    get_user_colour,
)

HELP_AP = (
    f"**Origin airport query**\nThe IATA, ICAO, name or id.\n"
    f"To specify multiple airports, separate them with a comma.\n"
    f"Learn more with `{cfg.bot.COMMAND_PREFIX}help airport`."
)
HELP_AC = (
    "**Aircraft query**\nThe short/full name of the aircraft (with custom engine/modifiers if necessary).\n"
    f"Learn more with `{cfg.bot.COMMAND_PREFIX}help aircraft`"
)
HELP_CONSTRAINT = (
    "**Constraint**\n"
    "- when not specified or given `NONE`, it'll optimise for max. profit per day per A/C\n"
    "- if a constraint is given, it'll optimise for max. profit per trip\n"
    "  - by default, it'll be interpreted as distance in kilometres (i.e. `16000` will return routes < 16,000km)\n"
    "  - to constrain by flight time instead, use the `HH:MM`, `1d, HH:MM` or "
    "[ISO 8601](https://en.wikipedia.org/wiki/ISO_8601#Durations) syntax"
)


def add_data(is_multi_hub: bool, o: Airport, d: Destination, is_cargo: bool, embed: discord.Embed):
    acr = d.ac_route

    profit_per_day_per_ac = acr.profit * acr.trips_per_day_per_ac
    origin_f = f"{format_ap_short(o, mode=0)}\n" if is_multi_hub else ""
    stopover_f = f"{format_ap_short(acr.stopover.airport, mode=1)}\n" if acr.stopover.exists else ""
    distance_f = f"{acr.stopover.full_distance if acr.stopover.exists else acr.route.direct_distance:.0f} km"
    flight_time_f = format_flight_time(acr.flight_time)
    num_ac_f = f"**__{acr.num_ac} ac__**" if acr.num_ac > 1 else f"{acr.num_ac} ac"
    embed.add_field(
        name=f"{origin_f}{stopover_f}{format_ap_short(d.airport, mode=2)}",
        value=(
            f"**Demand**: {format_demand(acr.route.pax_demand, is_cargo)}\n"
            f"**  Config**: {format_config(acr.config)}\n"
            f"**  Tickets**: {format_ticket(acr.ticket)}\n"
            f"** Details**: {distance_f} ({flight_time_f}), C$ {acr.contribution:.1f}/t\n"
            f"     {acr.trips_per_day_per_ac} t/d/ac × {num_ac_f}\n"
            f"** Profit**: $ {acr.profit:,.0f}/t, $ {profit_per_day_per_ac:,.0f}/d/ac\n"
        ),
        inline=False,
    )


class ButtonHandler(discord.ui.View):
    def __init__(
        self,
        message: discord.Message,
        is_multi_hub: bool,
        destinations: list[Destination],
        cols: dict[str, list],
        is_cargo: bool,
        file_suffix: str,
        user: User,
    ):
        super().__init__(timeout=15)
        self.message = message
        self.root_message = message
        self.start = 3

        if len(destinations) <= 3:
            self.handle_show_more.disabled = True
        self.is_multi_hub = is_multi_hub
        self.destinations = destinations
        self.cols = cols
        self.is_cargo = is_cargo
        self.file_suffix = file_suffix
        self.user = user

    async def on_timeout(self) -> None:
        self.clear_items()
        c = {"content": f"Go back to top: {self.root_message.jump_url}"} if self.start > 3 else {}
        await self.message.edit(view=None, **c)

    @discord.ui.button(label="Show more", style=discord.ButtonStyle.primary)
    async def handle_show_more(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(view=None)
        end = min(self.start + 3, len(self.destinations))
        emb = discord.Embed(
            colour=get_user_colour(self.user),
        )
        for d in self.destinations[self.start : end]:
            add_data(self.is_multi_hub, d.origin, d, self.is_cargo, emb)
        emb.set_footer(text=f"Showing {self.start}-{end} of {len(self.destinations)} routes")

        v = {"view": self} if self.start + 3 <= len(self.destinations) else {}
        self.message = await interaction.followup.send(embed=emb, **v)
        self.start += 3

    @discord.ui.button(label="CSV", emoji=ICSV)
    async def handle_export_csv(self, interaction: discord.Interaction, button: discord.ui.Button):
        button.disabled = True
        await interaction.response.edit_message(view=self)
        table = pa.Table.from_pydict({k[3:]: v for k, v in self.cols.items() if not (k.startswith("9"))})

        buf = io.BytesIO()
        csv.write_csv(table, buf)
        buf.seek(0)
        msg = await interaction.followup.send("Uploading...", wait=True)
        await msg.edit(
            content=None,
            attachments=[discord.File(buf, filename=f"routes_{self.file_suffix}.csv")],
        )

    @discord.ui.button(label="JSON", emoji=IJSON)
    async def handle_export_json(self, interaction: discord.Interaction, button: discord.ui.Button):
        button.disabled = True
        await interaction.response.edit_message(view=self)
        data = [d.to_dict(include_origin=self.is_multi_hub) for d in self.destinations]
        buf = io.BytesIO(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        buf.seek(0)
        msg = await interaction.followup.send("Uploading...", wait=True)
        await msg.edit(
            content=None,
            attachments=[discord.File(buf, filename=f"routes_{self.file_suffix}.json")],
        )
        buf.close()


class RoutesCog(BaseCog):
    def __init__(self, bot: commands.Bot, mpl_map: MPLMap):
        super().__init__(bot)
        self.executor = ThreadPoolExecutor(max_workers=4)
        # matplotlib is not thread-safe
        # run in one thread only to avoid the background colours getting messed up
        self.mpl_map_executor = ProcessPoolExecutor(max_workers=1)
        self.mpl_map = mpl_map

    @commands.command(
        brief="Searches best routes from a hub",
        help=(
            "The simplest way to get started is:```php\n"
            f"{cfg.bot.COMMAND_PREFIX}routes hkg a388\n"
            "```means: find the best routes departing `HKG` using `A380-800` (sort by highest profit *per trip*)."
            "But this **does not guarantee the best profit *per day***.\n"
            "Say you would like to follow a schedule of departing 3x per day instead: ```php\n"
            f"{cfg.bot.COMMAND_PREFIX}routes hkg a388 none 3\n"
            "```means: no constraints, find routes as long as I can depart it 3x per day "
            "(sort by highest profit *per aircraft per day*)\n"
        ),
        ignore_extra=False,
    )
    async def routes(
        self,
        ctx: commands.Context,
        ap_queries: list[Airport.SearchResult] = commands.parameter(converter=MultiAirportCvtr, description=HELP_AP),
        ac_query: Aircraft.SearchResult = commands.parameter(converter=AircraftCvtr, description=HELP_AC),
        constraint: tuple[float | None, float | None] = commands.parameter(
            converter=ConstraintCvtr,
            default=ConstraintCvtr._default,
            displayed_default="NONE",
            description=HELP_CONSTRAINT,
        ),
        trips_per_day_per_ac: tuple[int | None, AircraftRoute.Options.TPDMode] = commands.parameter(
            converter=TPDCvtr, default=TPDCvtr._default, displayed_default="AUTO", description=HELP_TPD
        ),
        config_algorithm: Aircraft.PaxConfig.Algorithm | Aircraft.CargoConfig.Algorithm = commands.parameter(
            converter=CfgAlgCvtr,
            default=CfgAlgCvtr._default,
            displayed_default="AUTO",
            description=HELP_CFG_ALG,
        ),
    ):
        is_cargo = ac_query.ac.type == Aircraft.Type.CARGO
        tpd, tpd_mode = trips_per_day_per_ac
        max_distance, max_flight_time = constraint
        cons_set = constraint != ConstraintCvtr._default
        tpd_set = trips_per_day_per_ac != TPDCvtr._default
        options = AircraftRoute.Options(
            **{
                k: v
                for k, v in {
                    "trips_per_day_per_ac": tpd,
                    "tpd_mode": tpd_mode,
                    "config_algorithm": config_algorithm,
                    "max_distance": max_distance,
                    "max_flight_time": max_flight_time,
                    "sort_by": (
                        AircraftRoute.Options.SortBy.PER_AC_PER_DAY
                        if cons_set
                        else AircraftRoute.Options.SortBy.PER_TRIP
                    ),
                }.items()
                if v is not None
            }
        )

        u, _ue = await fetch_user_info(ctx)
        if u.game_mode == u.GameMode.REALISM:
            for ap in ap_queries:
                if (warning := get_realism_departure_runway_warning(ap.ap, ac_query.ac)) is not None:
                    await ctx.send(embed=warning)
        # if the tpd is not provided, show generic warning of low tpd
        # otherwise, check if the constraint's equivalent flight time and tpd multiply to be <24 and ~24
        if cons_set:
            await self.check_constraints(ctx, ac_query, tpd, max_distance, max_flight_time, tpd_set, u.game_mode)

        rs = RoutesSearch([ap.ap for ap in ap_queries], ac_query.ac, options, u)

        loop = asyncio.get_event_loop()
        t_start = time.time()
        destinations = await loop.run_in_executor(self.executor, rs.get)
        t_end = time.time()

        if is_multi_hub := (len(ap_queries) > 1):
            title = f"Routes from {len(ap_queries)} hubs"
        else:
            title = format_ap_short(ap_queries[0].ap, mode=0)

        embed = discord.Embed(
            title=title,
            colour=get_user_colour(u),
        )
        profits = []  # each entry represents one aircraft
        if destinations:
            for i, d in enumerate(destinations):
                acr = d.ac_route

                profit_per_day_per_ac = acr.profit * acr.trips_per_day_per_ac
                for _ in range(acr.num_ac):
                    profits.append(profit_per_day_per_ac)
                if i > 2:
                    continue

                add_data(is_multi_hub, d.origin, d, is_cargo, embed)
        else:
            embed.description = (
                "There are no profitable routes found. Try relaxing the constraints or reducing the trips per day."
            )

        sorted_by = f" (sorted by $ {'per ac per day' if cons_set else 'per trip'})"
        embed.set_footer(
            text=(
                f"{len(destinations)} routes found in {(t_end - t_start) * 1000:.2f} ms{sorted_by}\n"
                f"top 10 ac: $ {sum(profits[:10]):,.0f}/d, 30 ac: $ {sum(profits[:30]):,.0f}/d\n"
                "Generating map and data..."
            ),
        )
        msg = await ctx.send(embed=embed)
        if not destinations:
            return

        cols = rs._get_columns(destinations, include_origin=is_multi_hub)
        iatas = "-".join([ap.ap.iata for ap in ap_queries])
        file_suffix = "_".join(
            [
                iatas,
                ac_query.ac.shortname,
                str(tpd),
            ]
        )
        btns = ButtonHandler(
            msg, is_multi_hub, destinations, cols, ac_query.ac.type == Aircraft.Type.CARGO, file_suffix, u
        )
        await msg.edit(view=btns)

        origin_lngs = [q.ap.lng for q in ap_queries]
        origin_lats = [q.ap.lat for q in ap_queries]
        im = await loop.run_in_executor(
            self.mpl_map_executor, self.mpl_map._plot_destinations, cols, origin_lngs, origin_lats
        )
        embed.set_image(url=f"attachment://routes_{file_suffix}.png")
        embed.set_footer(text="\n".join(embed.footer.text.split("\n")[:-1]))
        await msg.edit(
            embed=embed,
            attachments=[
                discord.File(im, filename=f"routes_{file_suffix}.png"),
            ],
        )

    async def check_constraints(
        self,
        ctx: commands.Context,
        ac_query: Aircraft.SearchResult,
        tpd: int | None,
        max_distance: float | None,
        max_flight_time: float | None,
        tpd_set: bool,
        game_mode: User.GameMode,
    ):
        cons_eq_t = (
            max_distance / ac_query.ac.speed / (1.5 if game_mode == User.GameMode.EASY else 1)
            if max_distance is not None
            else max_flight_time
        )
        cons_eq_f = ("equivalent to " if max_distance else "") + f"max `{cons_eq_t:.2f}` hr"
        sugg_cons_t, sugg_tpd = 24 / tpd, math.floor(24 / cons_eq_t)
        if (t_ttl := cons_eq_t * tpd) > 24 and tpd_set:
            await ctx.send(
                embed=discord.Embed(
                    title="Warning: Over-constrained!",
                    description=(
                        f"You have provided a constraint ({cons_eq_f}) and trips per day per A/C (`{tpd}`).\n"
                        f"But it is impossible to fly `{t_ttl:.2f}` hr in a day.\n"
                        f"I'll still respect your choice of `{tpd}` trips per day per A/C, but do note that the "
                        "suggested routes **may require you to depart very frequently**.\n\n"
                        f"To fix this, reduce your trips per day per A/C to `{sugg_tpd:.0f}`, or "
                        f"reduce your constraint to `{format_flight_time(sugg_cons_t, short=True)}` "
                        f"(`{ac_query.ac.speed * sugg_cons_t:.0f}` km)."
                    ),
                    color=COLOUR_WARNING,
                )
            )
        elif t_ttl < 24 * 0.9 and tpd_set:
            sugg_tpd_f = f"increase your trips per day per A/C to `{sugg_tpd:.0f}`, or " if sugg_tpd != tpd else ""
            await ctx.send(
                embed=discord.Embed(
                    title="Warning: Under-constrained!",
                    description=(
                        f"You have provided a constraint ({cons_eq_f}) and trips per day per A/C (`{tpd}`), "
                        f"meaning the average aircraft flies `{t_ttl:.2f}` hr in a day.\n"
                        "The profit per day per aircraft will be lower than the theoretical optimum.\n\n"
                        f"To fix this, {sugg_tpd_f}"
                        f"increase your constraint to `{format_flight_time(sugg_cons_t, short=True)}` "
                        f"(`{ac_query.ac.speed * sugg_cons_t:.0f}` km)."
                    ),
                    color=COLOUR_WARNING,
                )
            )

        if not tpd_set:
            await ctx.send(
                embed=discord.Embed(
                    title="Warning: Very short routes incoming!",
                    description=(
                        f"You have set a constraint ({cons_eq_f}), but did not set the trips per day per A/C.\n\n"
                        "I'll be sorting the routes by *max profit per day per A/C*, which will very likely "
                        "to be **extremely short routes**. You may not actually be able to depart that frequently, "
                        f"so I'd suggest you to specify the trips per day per aircraft (recommended: `{sugg_tpd}`)."
                        f"\n\nTip: Look at the tradeoff in the bottom right graph."
                    ),
                    color=COLOUR_WARNING,
                )
            )

    @routes.error
    async def route_error(self, ctx: commands.Context, error: commands.CommandError):
        h = CustomErrHandler(ctx, error, "routes")
        await h.invalid_airport()
        await h.too_many_airports()
        await h.invalid_aircraft()
        await h.invalid_tpd()
        await h.invalid_cfg_alg()
        await h.invalid_constraint()

        await h.banned_user()
        await h.too_many_args("argument")
        await h.common_mistakes()
        await h.raise_for_unhandled()
