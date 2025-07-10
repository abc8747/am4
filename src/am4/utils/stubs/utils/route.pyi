from __future__ import annotations
import am4.utils.aircraft
import am4.utils.airport
import am4.utils.demand
import am4.utils.game
import am4.utils.ticket
import typing
__all__ = ['AircraftRoute', 'Destination', 'Route', 'RoutesSearch', 'SameOdException']
class AircraftRoute:
    class Options:
        class SortBy:
            """
            Members:
            
              PER_TRIP
            
              PER_AC_PER_DAY
            """
            PER_AC_PER_DAY: typing.ClassVar[AircraftRoute.Options.SortBy]  # value = <SortBy.PER_AC_PER_DAY: 1>
            PER_TRIP: typing.ClassVar[AircraftRoute.Options.SortBy]  # value = <SortBy.PER_TRIP: 0>
            __members__: typing.ClassVar[dict[str, AircraftRoute.Options.SortBy]]  # value = {'PER_TRIP': <SortBy.PER_TRIP: 0>, 'PER_AC_PER_DAY': <SortBy.PER_AC_PER_DAY: 1>}
            def __eq__(self, other: typing.Any) -> bool:
                ...
            def __getstate__(self) -> int:
                ...
            def __hash__(self) -> int:
                ...
            def __index__(self) -> int:
                ...
            def __init__(self, value: int) -> None:
                ...
            def __int__(self) -> int:
                ...
            def __ne__(self, other: typing.Any) -> bool:
                ...
            def __repr__(self) -> str:
                ...
            def __setstate__(self, state: int) -> None:
                ...
            def __str__(self) -> str:
                ...
            @property
            def name(self) -> str:
                ...
            @property
            def value(self) -> int:
                ...
        class TPDMode:
            """
            Members:
            
              AUTO
            
              STRICT_ALLOW_MULTIPLE_AC
            
              STRICT
            """
            AUTO: typing.ClassVar[AircraftRoute.Options.TPDMode]  # value = <TPDMode.AUTO: 0>
            STRICT: typing.ClassVar[AircraftRoute.Options.TPDMode]  # value = <TPDMode.STRICT: 2>
            STRICT_ALLOW_MULTIPLE_AC: typing.ClassVar[AircraftRoute.Options.TPDMode]  # value = <TPDMode.STRICT_ALLOW_MULTIPLE_AC: 1>
            __members__: typing.ClassVar[dict[str, AircraftRoute.Options.TPDMode]]  # value = {'AUTO': <TPDMode.AUTO: 0>, 'STRICT_ALLOW_MULTIPLE_AC': <TPDMode.STRICT_ALLOW_MULTIPLE_AC: 1>, 'STRICT': <TPDMode.STRICT: 2>}
            def __eq__(self, other: typing.Any) -> bool:
                ...
            def __getstate__(self) -> int:
                ...
            def __hash__(self) -> int:
                ...
            def __index__(self) -> int:
                ...
            def __init__(self, value: int) -> None:
                ...
            def __int__(self) -> int:
                ...
            def __ne__(self, other: typing.Any) -> bool:
                ...
            def __repr__(self) -> str:
                ...
            def __setstate__(self, state: int) -> None:
                ...
            def __str__(self) -> str:
                ...
            @property
            def name(self) -> str:
                ...
            @property
            def value(self) -> int:
                ...
        config_algorithm: None | am4.utils.aircraft.Aircraft.PaxConfig.Algorithm | am4.utils.aircraft.Aircraft.CargoConfig.Algorithm
        max_distance: float
        max_flight_time: float
        min_distance: float
        min_flight_time: float
        sort_by: AircraftRoute.Options.SortBy
        tpd_mode: AircraftRoute.Options.TPDMode
        trips_per_day_per_ac: int
        def __init__(self, tpd_mode: AircraftRoute.Options.TPDMode = TPDMode.AUTO, trips_per_day_per_ac: int = 1, max_distance: float = 20015.086796020572, min_distance: float = 0.0, max_flight_time: float = 24.0, min_flight_time: float = 0.0, config_algorithm: None | am4.utils.aircraft.Aircraft.PaxConfig.Algorithm | am4.utils.aircraft.Aircraft.CargoConfig.Algorithm = None, sort_by: AircraftRoute.Options.SortBy = SortBy.PER_TRIP) -> None:
            ...
    class Stopover:
        @staticmethod
        def find_by_efficiency(origin: am4.utils.airport.Airport, destination: am4.utils.airport.Airport, aircraft: am4.utils.aircraft.Aircraft, game_mode: am4.utils.game.User.GameMode) -> AircraftRoute.Stopover:
            ...
        def __repr__(self) -> str:
            ...
        def to_dict(self) -> dict:
            ...
        @property
        def airport(self) -> am4.utils.airport.Airport:
            ...
        @property
        def exists(self) -> bool:
            ...
        @property
        def full_distance(self) -> float:
            ...
    class Warning:
        """
        Members:
        
          ERR_RWY_TOO_SHORT
        
          ERR_DISTANCE_ABOVE_SPECIFIED
        
          ERR_DISTANCE_BELOW_SPECIFIED
        
          ERR_DISTANCE_TOO_LONG
        
          ERR_DISTANCE_TOO_SHORT
        
          REDUCED_CONTRIBUTION
        
          ERR_NO_STOPOVER
        
          ERR_FLIGHT_TIME_ABOVE_SPECIFIED
        
          ERR_FLIGHT_TIME_BELOW_SPECIFIED
        
          ERR_INSUFFICIENT_DEMAND
        
          ERR_TRIPS_PER_DAY_TOO_HIGH
        """
        ERR_DISTANCE_ABOVE_SPECIFIED: typing.ClassVar[AircraftRoute.Warning]  # value = <Warning.ERR_DISTANCE_ABOVE_SPECIFIED: 1>
        ERR_DISTANCE_BELOW_SPECIFIED: typing.ClassVar[AircraftRoute.Warning]  # value = <Warning.ERR_DISTANCE_BELOW_SPECIFIED: 2>
        ERR_DISTANCE_TOO_LONG: typing.ClassVar[AircraftRoute.Warning]  # value = <Warning.ERR_DISTANCE_TOO_LONG: 3>
        ERR_DISTANCE_TOO_SHORT: typing.ClassVar[AircraftRoute.Warning]  # value = <Warning.ERR_DISTANCE_TOO_SHORT: 4>
        ERR_FLIGHT_TIME_ABOVE_SPECIFIED: typing.ClassVar[AircraftRoute.Warning]  # value = <Warning.ERR_FLIGHT_TIME_ABOVE_SPECIFIED: 7>
        ERR_FLIGHT_TIME_BELOW_SPECIFIED: typing.ClassVar[AircraftRoute.Warning]  # value = <Warning.ERR_FLIGHT_TIME_BELOW_SPECIFIED: 8>
        ERR_INSUFFICIENT_DEMAND: typing.ClassVar[AircraftRoute.Warning]  # value = <Warning.ERR_INSUFFICIENT_DEMAND: 9>
        ERR_NO_STOPOVER: typing.ClassVar[AircraftRoute.Warning]  # value = <Warning.ERR_NO_STOPOVER: 6>
        ERR_RWY_TOO_SHORT: typing.ClassVar[AircraftRoute.Warning]  # value = <Warning.ERR_RWY_TOO_SHORT: 0>
        ERR_TRIPS_PER_DAY_TOO_HIGH: typing.ClassVar[AircraftRoute.Warning]  # value = <Warning.ERR_TRIPS_PER_DAY_TOO_HIGH: 10>
        REDUCED_CONTRIBUTION: typing.ClassVar[AircraftRoute.Warning]  # value = <Warning.REDUCED_CONTRIBUTION: 5>
        __members__: typing.ClassVar[dict[str, AircraftRoute.Warning]]  # value = {'ERR_RWY_TOO_SHORT': <Warning.ERR_RWY_TOO_SHORT: 0>, 'ERR_DISTANCE_ABOVE_SPECIFIED': <Warning.ERR_DISTANCE_ABOVE_SPECIFIED: 1>, 'ERR_DISTANCE_BELOW_SPECIFIED': <Warning.ERR_DISTANCE_BELOW_SPECIFIED: 2>, 'ERR_DISTANCE_TOO_LONG': <Warning.ERR_DISTANCE_TOO_LONG: 3>, 'ERR_DISTANCE_TOO_SHORT': <Warning.ERR_DISTANCE_TOO_SHORT: 4>, 'REDUCED_CONTRIBUTION': <Warning.REDUCED_CONTRIBUTION: 5>, 'ERR_NO_STOPOVER': <Warning.ERR_NO_STOPOVER: 6>, 'ERR_FLIGHT_TIME_ABOVE_SPECIFIED': <Warning.ERR_FLIGHT_TIME_ABOVE_SPECIFIED: 7>, 'ERR_FLIGHT_TIME_BELOW_SPECIFIED': <Warning.ERR_FLIGHT_TIME_BELOW_SPECIFIED: 8>, 'ERR_INSUFFICIENT_DEMAND': <Warning.ERR_INSUFFICIENT_DEMAND: 9>, 'ERR_TRIPS_PER_DAY_TOO_HIGH': <Warning.ERR_TRIPS_PER_DAY_TOO_HIGH: 10>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: int) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        def to_str(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    @staticmethod
    @typing.overload
    def calc_co2(ac: am4.utils.aircraft.Aircraft, cfg: am4.utils.aircraft.Aircraft.PaxConfig, distance: float, user: am4.utils.game.User = am4.utils.game.User.Default(), ci: int = 200) -> float:
        ...
    @staticmethod
    @typing.overload
    def calc_co2(ac: am4.utils.aircraft.Aircraft, cfg: am4.utils.aircraft.Aircraft.CargoConfig, distance: float, user: am4.utils.game.User = am4.utils.game.User.Default(), ci: int = 200) -> float:
        ...
    @staticmethod
    def calc_fuel(ac: am4.utils.aircraft.Aircraft, distance: float, user: am4.utils.game.User = am4.utils.game.User.Default(), ci: int = 200) -> float:
        ...
    @staticmethod
    def create(ap0: am4.utils.airport.Airport, ap1: am4.utils.airport.Airport, ac: am4.utils.aircraft.Aircraft, options: AircraftRoute.Options = AircraftRoute.Options(), user: am4.utils.game.User = am4.utils.game.User.Default()) -> AircraftRoute:
        ...
    @staticmethod
    def estimate_load(reputation: float = 87, autoprice_ratio: float = 1.06, has_stopover: bool = False) -> float:
        ...
    def __repr__(self) -> str:
        ...
    def to_dict(self) -> dict:
        ...
    @property
    def acheck_cost(self) -> float:
        ...
    @property
    def ci(self) -> int:
        ...
    @property
    def co2(self) -> float:
        ...
    @property
    def config(self) -> am4.utils.aircraft.Aircraft.PaxConfig | am4.utils.aircraft.Aircraft.CargoConfig:
        ...
    @property
    def contribution(self) -> float:
        ...
    @property
    def flight_time(self) -> float:
        ...
    @property
    def fuel(self) -> float:
        ...
    @property
    def income(self) -> float:
        ...
    @property
    def max_income(self) -> float:
        ...
    @property
    def max_tpd(self) -> int | None:
        ...
    @property
    def needs_stopover(self) -> bool:
        ...
    @property
    def num_ac(self) -> int:
        ...
    @property
    def profit(self) -> float:
        ...
    @property
    def repair_cost(self) -> float:
        ...
    @property
    def route(self) -> Route:
        ...
    @property
    def stopover(self) -> AircraftRoute.Stopover:
        ...
    @property
    def ticket(self) -> am4.utils.ticket.PaxTicket | am4.utils.ticket.CargoTicket | am4.utils.ticket.VIPTicket:
        ...
    @property
    def trips_per_day_per_ac(self) -> int:
        ...
    @property
    def valid(self) -> bool:
        ...
    @property
    def warnings(self) -> list[AircraftRoute.Warning]:
        ...
class Destination:
    def to_dict(self, include_origin: bool = False) -> dict:
        ...
    @property
    def ac_route(self) -> AircraftRoute:
        ...
    @property
    def airport(self) -> am4.utils.airport.Airport:
        ...
    @property
    def origin(self) -> am4.utils.airport.Airport:
        ...
class Route:
    @staticmethod
    @typing.overload
    def calc_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        ...
    @staticmethod
    @typing.overload
    def calc_distance(a0: am4.utils.airport.Airport, a1: am4.utils.airport.Airport) -> float:
        ...
    @staticmethod
    def create(ap0: am4.utils.airport.Airport, ap1: am4.utils.airport.Airport) -> Route:
        ...
    def __repr__(self) -> str:
        ...
    def to_dict(self) -> dict:
        ...
    @property
    def direct_distance(self) -> float:
        ...
    @property
    def pax_demand(self) -> am4.utils.demand.PaxDemand:
        ...
    @property
    def valid(self) -> bool:
        ...
class RoutesSearch:
    def __init__(self, ap0: list[am4.utils.airport.Airport], ac: am4.utils.aircraft.Aircraft, options: AircraftRoute.Options = AircraftRoute.Options(), user: am4.utils.game.User = am4.utils.game.User.Default()) -> None:
        ...
    def _get_columns(self, dests: list[Destination], include_origin: bool = False) -> dict[str, list]:
        ...
    def get(self) -> list[Destination]:
        ...
class SameOdException(Exception):
    pass
