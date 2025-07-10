from datetime import timedelta
from typing import Annotated, Any, List, Literal, NamedTuple

from discord.ext import commands
from pydantic import BaseModel, Field
from pydantic_core import PydanticCustomError, ValidationError

from am4.utils.aircraft import Aircraft
from am4.utils.airport import Airport
from am4.utils.route import AircraftRoute

from ..db.models.aircraft import PyConfigAlgorithmCargo, PyConfigAlgorithmPax
from ..db.models.game import PyUser
from ..db.models.route import (
    PyACROptionsMaxDistance,
    PyACROptionsTPDMode,
    PyACROptionsTripsPerDayPerAC,
)
from .errors import (
    AircraftNotFoundError,
    AirportNotFoundError,
    CfgAlgValidationError,
    ConstraintValidationError,
    PriceValidationError,
    SettingValueValidationError,
    TooManyAirportsError,
    TPDValidationError,
)


class AirportCvtr(commands.Converter):
    async def convert(self, ctx: commands.Context, query: str) -> Airport.SearchResult:
        acsr = Airport.search(query)

        if not acsr.ap.valid:
            raise AirportNotFoundError(acsr)
        return acsr


class MultiAirportCvtr(commands.Converter):
    async def convert(self, ctx: commands.Context, query: str) -> List[Airport.SearchResult]:
        queries = query.split(",")
        MAX_AIRPORTS = 24  # prevent DoS attacks & stay within upload limits
        if (num_airports := len(queries)) > MAX_AIRPORTS:
            raise TooManyAirportsError(num_airports, max_airports=MAX_AIRPORTS)

        acsr_list = []
        for q in queries:
            acsr = Airport.search(q)
            if not acsr.ap.valid:
                raise AirportNotFoundError(acsr)
            acsr_list.append(acsr)
        # TODO: handle duplicate airports?

        return acsr_list


class AircraftCvtr(commands.Converter):
    async def convert(self, ctx: commands.Context, query: str) -> Aircraft.SearchResult:
        acsr = Aircraft.search(query)

        if not acsr.ac.valid:
            raise AircraftNotFoundError(acsr)
        return acsr


class _ACROptions(BaseModel):
    algorithm_pax: PyConfigAlgorithmPax
    algorithm_cargo: PyConfigAlgorithmCargo
    max_distance: PyACROptionsMaxDistance
    max_flight_time: Annotated[timedelta, Field(gt=0, lt=72 * 3600)]
    __tpd_mode: PyACROptionsTPDMode
    trips_per_day_per_ac: PyACROptionsTripsPerDayPerAC


class _SettingKeyCustom(BaseModel):
    training: Literal["max", "min"]


def acro_cast(k: str, v: Any) -> _ACROptions:
    return _ACROptions.__pydantic_validator__.validate_assignment(_ACROptions.model_construct(), k, v)


class SettingValueCvtr(commands.Converter):
    async def convert(self, ctx: commands.Context, value: Any) -> Any:
        key = ctx.args[-1]  # TODO: this is a hack, find a better way to get the key!
        model = _SettingKeyCustom if key == "training" else PyUser
        try:
            u_new = model.__pydantic_validator__.validate_assignment(model.model_construct(), key, value)
        except ValidationError as err:
            if key == "load" or key == "cargo_load":
                err.errors()[0]["msg"] += ". Load factor must be a percentage (e.g., `87%`)."
            raise SettingValueValidationError(err)
        v_new = getattr(u_new, key)
        return v_new


class TPDCvtr(commands.Converter):
    _default = (1, AircraftRoute.Options.TPDMode.AUTO)

    async def convert(self, ctx: commands.Context, tpdo: str) -> tuple[int | None, AircraftRoute.Options.TPDMode]:
        if tpdo is None or (tpd := tpdo.strip().lower()) == "auto":
            return self._default
        strict = tpd.endswith("!")
        try:
            return (
                acro_cast("trips_per_day_per_ac", tpd[:-1] if strict else tpd).trips_per_day_per_ac,
                AircraftRoute.Options.TPDMode.STRICT
                if strict
                else AircraftRoute.Options.TPDMode.STRICT_ALLOW_MULTIPLE_AC,
            )
        except ValidationError as e:
            raise TPDValidationError(e)


class CfgAlgCvtr(commands.Converter):
    async def convert(
        self, ctx: commands.Context, alg: str
    ) -> Aircraft.PaxConfig.Algorithm | Aircraft.CargoConfig.Algorithm:
        ac: Aircraft.SearchResult = next(a for a in ctx.args if isinstance(a, Aircraft.SearchResult))
        field_name = "algorithm_cargo" if ac.ac.type == Aircraft.Type.CARGO else "algorithm_pax"
        try:
            alg_parsed = getattr(acro_cast(field_name, alg.upper()), field_name)
        except ValidationError as e:
            raise CfgAlgValidationError(e)

        return (
            Aircraft.CargoConfig.Algorithm.__members__.get(alg_parsed)
            if ac.ac.type == Aircraft.Type.CARGO
            else Aircraft.PaxConfig.Algorithm.__members__.get(alg_parsed)
        )

    @staticmethod
    async def _default(ctx: commands.Context) -> Aircraft.PaxConfig.Algorithm | Aircraft.CargoConfig.Algorithm:
        try:
            ac: Aircraft.SearchResult = next(a for a in ctx.args if isinstance(a, Aircraft.SearchResult))
        except StopIteration:
            # just to handle empty aircraft slots
            return Aircraft.PaxConfig.Algorithm.AUTO
        return (
            Aircraft.CargoConfig.Algorithm.AUTO
            if ac.ac.type == Aircraft.Type.CARGO
            else Aircraft.PaxConfig.Algorithm.AUTO
        )


class Constraint(NamedTuple):
    min_distance: float | None
    max_distance: float | None
    min_flight_time: float | None
    max_flight_time: float | None


class ConstraintCvtr(commands.Converter):
    _default = Constraint(None, None, None, None)

    def _parse_one(self, value: str) -> tuple[float | None, float | None]:
        try:
            dist_parsed = acro_cast("max_distance", value).max_distance
            return dist_parsed, None
        except ValidationError:
            try:
                time_parsed = acro_cast("max_flight_time", value).max_flight_time
                return None, time_parsed.total_seconds() / 3600
            except ValidationError as e:
                raise ConstraintValidationError(e)

    async def convert(self, ctx: commands.Context, constraint: str) -> Constraint:
        constraint = constraint.strip().lower()
        if constraint == "none":
            return self._default

        separator = ".."
        if separator in constraint:
            parts = [p.strip() for p in constraint.split(separator, 1)]
            min_val_str, max_val_str = parts[0] or None, parts[1] or None
        else:
            min_val_str, max_val_str = None, constraint

        min_dist, min_time = self._parse_one(min_val_str) if min_val_str else (None, None)
        max_dist, max_time = self._parse_one(max_val_str) if max_val_str else (None, None)

        return Constraint(min_dist, max_dist, min_time, max_time)


class _Price(BaseModel):
    fuel: Annotated[float, Field(gt=0, le=900)]
    co2: Annotated[float, Field(gt=0, le=140)]


class PriceCvtr(commands.Converter):
    async def convert(self, ctx: commands.Context, price: str | None) -> tuple[Literal["Fuel", "CO₂"], float] | None:
        if price is None:
            return None
        if len(price) < 2:
            raise PriceValidationError(
                PydanticCustomError(
                    "missing_price",
                    "Cannot find the price type or value.",
                )
            )
        allowed = {
            "f": "fuel",
            "c": "co2",
        }
        k, v = allowed.get(price[0].lower(), None), price[1:]
        if k is None:
            raise PriceValidationError(
                PydanticCustomError(
                    "invalid_price_type",
                    "The price type `{price_type}` is invalid. Start with `f` for fuel or `c` for CO₂.",
                    {"price_type": price[0].lower()},
                )
            )
        try:
            v_new = _Price.__pydantic_validator__.validate_assignment(_Price.model_construct(), k, v)
        except ValidationError as e:
            raise PriceValidationError(e)
        k_formatted = {
            "fuel": "Fuel",
            "co2": "CO₂",
        }
        return k_formatted.get(k), getattr(v_new, k)
