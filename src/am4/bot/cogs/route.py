import discord
from discord.ext import commands

from am4.utils.aircraft import Aircraft
from am4.utils.airport import Airport
from am4.utils.route import AircraftRoute, Route, SameOdException

from ...config import cfg
from ..base import BaseCog
from ..converters import AircraftCvtr, AirportCvtr, CfgAlgCvtr, TPDCvtr
from ..errors import CustomErrHandler
from ..utils import (
    COLOUR_ERROR,
    COLOUR_GENERIC,
    HELP_CFG_ALG,
    HELP_TPD,
    fetch_user_info,
    format_ap_short,
    format_config,
    format_demand,
    format_flight_time,
    format_ticket,
    get_realism_departure_runway_warning,
    get_user_colour,
)

HELP_AP_ARG0 = "**Origin airport query**\nLearn more using `$help airport`."
HELP_AP_ARG1 = "**Destination airport query**\nLearn more using `$help airport`."
HELP_AC_ARG0 = "**Aircraft query**\nLearn more about how to customise engine/modifiers using `$help aircraft`."


def format_additional(
    income: float,
    fuel: float,
    fuel_price: float,
    co2: float,
    co2_price: float,
    acheck_cost: float,
    repair_cost: float,
    profit: float,
    contribution: float,
    ci: float,
):
    return (
        f"**   Income**: $ {income:,.0f}\n"
        f"**  -    Fuel**: $ {fuel * fuel_price / 1000:,.0f} ({fuel:,.0f} lb)\n"
        f"**  -     CO₂**: $ {co2 * co2_price / 1000:,.0f} ({co2:,.0f} q)\n"
        f"**  - Acheck**: $ {acheck_cost:,.0f}\n"
        f"**  -   Repair**: $ {repair_cost:,.0f}\n"
        f"**  =   Profit**: $ {profit:,.0f}\n"
        f"**  Contrib**: $ {contribution:.2f} (CI={ci})\n"
    )


class RouteCog(BaseCog):
    @commands.command(
        brief="Finds information about a route",
        help=(
            "Finds information about an route given an origin and destination (and optionally the aircraft), examples:```php\n"
            f"{cfg.bot.COMMAND_PREFIX}route hkg lhr\n"
            f"{cfg.bot.COMMAND_PREFIX}route id:3500 egll\n"
            f"{cfg.bot.COMMAND_PREFIX}route vhhh tpe dc910\n"
            f"{cfg.bot.COMMAND_PREFIX}route hkg yvr a388[sfc] 3\n"
            "```"
        ),
        ignore_extra=False,
    )
    async def route(
        self,
        ctx: commands.Context,
        ap0_query: Airport.SearchResult = commands.parameter(converter=AirportCvtr, description=HELP_AP_ARG0),
        ap1_query: Airport.SearchResult = commands.parameter(converter=AirportCvtr, description=HELP_AP_ARG1),
        ac_query: Aircraft.SearchResult | None = commands.parameter(
            converter=AircraftCvtr, default=None, description=HELP_AC_ARG0
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
        if ac_query is None:
            try:
                r = Route.create(ap0_query.ap, ap1_query.ap)
            except SameOdException as e:
                embed = discord.Embed(
                    title="Invalid route!",
                    description=str(e),
                    colour=COLOUR_ERROR,
                )
                await ctx.send(embed=embed)
                return
            embed = self.get_basic_route_embed(ap0_query, ap1_query, r)
            await ctx.send(embed=embed)
            return
        is_cargo = ac_query.ac.type == Aircraft.Type.CARGO
        tpd, tpd_mode = trips_per_day_per_ac

        options = AircraftRoute.Options(trips_per_day_per_ac=tpd, tpd_mode=tpd_mode, config_algorithm=config_algorithm)
        u, _ue = await fetch_user_info(ctx)
        if (
            u.game_mode == u.GameMode.REALISM
            and (warning := get_realism_departure_runway_warning(ac_query.ac, (ap0_query.ap,))) is not None
        ):
            await ctx.send(embed=warning)

        acr = AircraftRoute.create(ap0_query.ap, ap1_query.ap, ac_query.ac, options, u)
        if not acr.valid:
            embed_w = discord.Embed(
                title="Route cannot be created.",
                description="\n".join(f"- {w.to_str()}" for w in acr.warnings) or "Unknown error.",
                colour=COLOUR_ERROR,
            )
            embed = self.get_basic_route_embed(ap0_query, ap1_query, acr.route)
            await ctx.send(embeds=[embed_w, embed])
            return

        sa = acr.stopover.airport
        stopover_f = f"{format_ap_short(sa, mode=1)}\n" if acr.stopover.exists else ""
        distance_f = (
            f"{acr.stopover.full_distance:.3f} km (+{acr.stopover.full_distance - acr.route.direct_distance:.3f} km)"
            if acr.stopover.exists
            else f"{acr.route.direct_distance:.3f} km (direct)"
        )
        description = (
            f"**Flight Time**: {format_flight_time(acr.flight_time)} ({acr.flight_time:.3f} hr)\n"
            f"**  Schedule**: {acr.trips_per_day_per_ac:.0f} trips/day/ac × {acr.num_ac} A/C needed\n"
            f"**  Demand**: {format_demand(acr.route.pax_demand, is_cargo)}\n"
            f"**  Config**: {format_config(acr.config)}\n"
            f"**   Tickets**: {format_ticket(acr.ticket)}\n"
            f"** Distance**: {distance_f}\n"
        )
        embed = discord.Embed(
            title=f"{format_ap_short(ap0_query.ap, mode=0)}\n{stopover_f}{format_ap_short(ap1_query.ap, mode=2)}",
            description=description,
            colour=get_user_colour(u),
        )
        embed.add_field(
            name="Per Trip",
            value=format_additional(
                income=acr.income,
                fuel=acr.fuel,
                fuel_price=u.fuel_price,
                co2=acr.co2,
                co2_price=u.co2_price,
                acheck_cost=acr.acheck_cost,
                repair_cost=acr.repair_cost,
                profit=acr.profit,
                contribution=acr.contribution,
                ci=acr.ci,
            ),
        )
        mul = acr.trips_per_day_per_ac
        embed.add_field(
            name="Per Day, Per Aircraft",
            value=format_additional(
                income=acr.income * mul,
                fuel=acr.fuel * mul,
                fuel_price=u.fuel_price,
                co2=acr.co2 * mul,
                co2_price=u.co2_price,
                acheck_cost=acr.acheck_cost * mul,
                repair_cost=acr.repair_cost * mul,
                profit=acr.profit * mul,
                contribution=acr.contribution * mul,
                ci=acr.ci,
            ),
        )
        await ctx.send(embed=embed)

    def get_basic_route_embed(self, ap0_query: Airport.SearchResult, ap1_query: Airport.SearchResult, r: Route):
        embed = discord.Embed(
            title=f"{format_ap_short(ap0_query.ap, mode=0)}\n{format_ap_short(ap1_query.ap, mode=2)}",
            description=(
                f"** Demand**: {format_demand(r.pax_demand)}\n"
                f"**     ** {format_demand(r.pax_demand, as_cargo=True)}\n"
                f"**Distance**: {r.direct_distance:.3f} km (direct)"
            ),
            colour=COLOUR_GENERIC,
        )

        return embed

    @route.error
    async def route_error(self, ctx: commands.Context, error: commands.CommandError):
        h = CustomErrHandler(ctx, error, "route")
        await h.invalid_airport(route_typo=True)
        await h.invalid_aircraft()
        await h.invalid_tpd()
        await h.invalid_cfg_alg()

        await h.banned_user()
        await h.too_many_args("argument")
        await h.common_mistakes()
        await h.raise_for_unhandled()
