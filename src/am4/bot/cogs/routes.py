from __future__ import annotations

import asyncio
import io
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

import discord
import orjson
import polars as pl
from discord.ext import commands

from am4.utils.aircraft import Aircraft
from am4.utils.airport import Airport
from am4.utils.game import User
from am4.utils.route import AircraftRoute, Destination, RoutesSearch

from ...config import cfg
from ..base import BaseCog
from ..converters import AircraftCvtr, CfgAlgCvtr, Constraint, ConstraintCvtr, MultiAirportCvtr, TPDCvtr
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
    ""
)
HELP_CONSTRAINT = (
    "**Constraint**\n"
    "Defines the search range for distance (km) or flight time (HH:MM, see "
    "[ISO 8601](https://en.wikipedia.org/wiki/ISO_8601#Durations) syntax).\n"
    "- `16000`: max distance of 16,000 km\n"
    "- `8000..16000`: distance between 8,000 and 16,000 km\n"
    "- `08:00..12:00`: flight time between 8 and 12 hours\n"
    "- `12:00..`: flight time of at least 12 hours\n"
    "- `..16000`: same as `16000`\n"
    "When a constraint is given, routes are sorted by profit per trip. Otherwise, by profit per day."
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
        mpl_map: MPLMap,
        mpl_map_executor: ProcessPoolExecutor,
    ):
        super().__init__(timeout=15)
        self.message = message
        self.root_message = message
        self.start = 3

        if len(destinations) <= 3:
            self.handle_show_more.disabled = True
        if not is_multi_hub:
            self.remove_item(self.handle_compare_hubs)

        self.is_multi_hub = is_multi_hub
        self.destinations = destinations
        self.cols = cols
        self.is_cargo = is_cargo
        self.file_suffix = file_suffix
        self.user = user
        self.mpl_map = mpl_map
        self.mpl_map_executor = mpl_map_executor

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
        df = pl.DataFrame({k[3:]: v for k, v in self.cols.items() if not (k.startswith("9"))})
        buf = io.BytesIO()
        df.write_csv(buf)
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

    @discord.ui.button(label="Compare Hubs", style=discord.ButtonStyle.secondary)
    async def handle_compare_hubs(self, interaction: discord.Interaction, button: discord.ui.Button):
        button.disabled = True
        await interaction.response.edit_message(view=self)

        hubs_data: dict[str, HubProfitData] = {}
        for d in self.destinations:
            origin_iata = d.origin.iata
            hub = hubs_data.setdefault(
                origin_iata,
                HubProfitData(profits_per_ac=[], hub_cost=d.origin.hub_cost),
            )

            profit_per_day = d.ac_route.profit * d.ac_route.trips_per_day_per_ac
            for _ in range(d.ac_route.num_ac):
                hub.profits_per_ac.append(profit_per_day)

        loop = asyncio.get_event_loop()
        im_buffer = await loop.run_in_executor(self.mpl_map_executor, self.mpl_map._plot_hub_comparison, hubs_data)

        await interaction.followup.send(file=discord.File(im_buffer, filename=f"hub_comparison_{self.file_suffix}.jpg"))


@dataclass
class HubProfitData:
    profits_per_ac: list[float]
    hub_cost: float


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
        constraint: Constraint = commands.parameter(
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
        cons_set = constraint != ConstraintCvtr._default
        tpd_set = trips_per_day_per_ac != TPDCvtr._default
        options = AircraftRoute.Options(
            **{
                k: v
                for k, v in {
                    "trips_per_day_per_ac": tpd,
                    "tpd_mode": tpd_mode,
                    "config_algorithm": config_algorithm,
                    "max_distance": constraint.max_distance,
                    "min_distance": constraint.min_distance,
                    "max_flight_time": constraint.max_flight_time,
                    "min_flight_time": constraint.min_flight_time,
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
        if (
            u.game_mode == u.GameMode.REALISM
            and (warning := get_realism_departure_runway_warning(ac_query.ac, [ap.ap for ap in ap_queries])) is not None
        ):
            await ctx.send(embed=warning)
        # if the tpd is not provided, show generic warning of low tpd
        # otherwise, check if the constraint's equivalent flight time and tpd multiply to be <24 and ~24
        if cons_set:
            await self.check_constraints(
                ctx, ac_query, tpd, constraint.max_distance, constraint.max_flight_time, tpd_set, u.game_mode
            )

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
            msg,
            is_multi_hub,
            destinations,
            cols,
            ac_query.ac.type == Aircraft.Type.CARGO,
            file_suffix,
            u,
            self.mpl_map,
            self.mpl_map_executor,
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
        cons_eq_f = (f"{max_distance}, equivalent to " if max_distance else "") + f"max `{cons_eq_t:.2f}` hr"
        sugg_cons_t, sugg_tpd = 24 / tpd, math.floor(24 / cons_eq_t)
        if (t_ttl := cons_eq_t * tpd) > 24 and tpd_set:
            embed = discord.Embed(
                title="Warning: Impossible schedule",
                description=(
                    f"The provided constraint ({cons_eq_f}) and trips per day (`{tpd}`) require a total flight time "
                    f"of `{t_ttl:.2f}` hours, which exceeds 24 hours.\n"
                    f"Routes will still respect the `{tpd}` trips per day constraint, but will "
                    "require very frequent departures."
                ),
                color=COLOUR_WARNING,
            )
            embed.add_field(
                name="help: to create a valid 24-hour schedule, try one of the following:",
                value=(
                    f"- reduce trips per day to `{sugg_tpd:.0f}`\n"
                    f"- reduce the constraint to `{format_flight_time(sugg_cons_t, short=True)}` "
                    f"(approx. `{ac_query.ac.speed * sugg_cons_t:.0f}` km)"
                ),
                inline=False,
            )
            await ctx.send(embed=embed)
        elif t_ttl < 24 * 0.9 and tpd_set:
            embed = discord.Embed(
                title="Warning: Inefficient schedule",
                description=(
                    f"The provided constraint ({cons_eq_f}) and trips per day (`{tpd}`) result in a total flight time "
                    f"of `{t_ttl:.2f}` hours, which underutilizes the aircraft."
                ),
                color=COLOUR_WARNING,
            )
            sugg_tpd_f = f"- increase trips per day to `{sugg_tpd:.0f}`\n" if sugg_tpd != tpd else ""
            embed.add_field(
                name="help: to maximize daily profit, consider one of the following",
                value=(
                    f"{sugg_tpd_f}"
                    f"- increase the constraint to `{format_flight_time(sugg_cons_t, short=True)}` "
                    f"(approx. `{ac_query.ac.speed * sugg_cons_t:.0f}` km)"
                ),
                inline=False,
            )
            await ctx.send(embed=embed)
        if not tpd_set:
            embed = discord.Embed(
                title="note: sorting by profit per day",
                description=(
                    f"A constraint ({cons_eq_f}) was provided without a specific number of trips per day.\n"
                    "Routes will be sorted by maximum profit per day, "
                    "which favors very short routes with high frequency."
                ),
                color=COLOUR_WARNING,
            )
            embed.add_field(
                name="help: for more realistic schedules, specify the number of trips per day",
                value=(
                    f"based on your constraint, a schedule of `{sugg_tpd}` trips per day would maximize flight hours."
                ),
                inline=False,
            )
            await ctx.send(embed=embed)

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
