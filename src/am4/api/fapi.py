import asyncio
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.responses import ORJSONResponse
from uvicorn import Config, Server

from am4.utils.aircraft import Aircraft
from am4.utils.airport import Airport
from am4.utils.db import init as utils_init
from am4.utils.route import AircraftRoute, Route, RoutesSearch

from ..config import cfg
from .models import (
    FAPIReqACROptions,
    FAPIReqACSearchQuery,
    FAPIReqAPSearchQuery,
    FAPIReqUser,
    FAPIRespACRoute,
    FAPIRespACRouteFind,
    FAPIRespAircraft,
    FAPIRespAircraftNotFound,
    FAPIRespAirport,
    FAPIRespAirportNotFound,
    FAPIRespRoute,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    utils_init()
    yield


app = FastAPI(
    title="AM4Tools V2 API (Alpha)",
    description="""A collection of calculators and tools for Airline Manager 4, aimed to facilitate statistical analyses and promote the development of third-party tools. This version is primarily rewritten in C++ for performance, in particular, route finding.

**This API is currently in alpha** - please report any issues on our [AM4Tools Discord server](https://discord.gg/4tVQHtf), or ping me at @cathayexpress if you encounter critical issues such as 500 Internal Server Error. This project is open source - feel free to open pull requests on [GitHub](https://github.com/abc8747/am4).

# Key features
- Fuzzy airport and aircraft search
- Basic route details (P2P)
- Advanced route details (P2P, with aircraft)
- Automatic best stopover, trips per day and seat algorithm selection
- Advanced route finder (P2P, with aircraft, origin hub, and filters) - **be careful: the massive response payload may crash this tab**: use Insomnia or Postman instead!
- Fuel, CO2, misc costs and contribution estimation

At this stage, **no authentication is required** to access the API. An API token in the future will be required as this scales.

This service is 100% free to use for all users.

Open one of the endpoints and click the try it out button right here in your browser, or download `openapi.json` to test it out!""",
    version="0.1.8",
    lifespan=lifespan,
)


def construct_acnf_response(param_name: str, ac_sugg: list[Aircraft.Suggestion]) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=404,
        content={
            "status": "not_found",
            "parameter": param_name,
            "suggestions": [{"aircraft": a0s.ac.to_dict(), "score": a0s.score} for a0s in ac_sugg],
        },
    )


@app.get(
    "/ac/search",
    response_model=FAPIRespAircraft,
    responses={404: {"model": FAPIRespAircraftNotFound}},
)
async def ac_search(query: FAPIReqACSearchQuery):
    ac = Aircraft.search(query)
    if ac.ac.valid:
        return {"status": "success", "aircraft": ac.ac.to_dict()}

    return construct_acnf_response("ac", Aircraft.suggest(ac.parse_result))


def construct_apnf_response(param_name: str, ap_sugg: list[Airport.Suggestion]) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=404,
        content={
            "status": "not_found",
            "parameter": param_name,
            "suggestions": [{"airport": a0s.ap.to_dict(), "score": a0s.score} for a0s in ap_sugg],
        },
    )


@app.get(
    "/ap/search",
    response_model=FAPIRespAirport,
    responses={404: {"model": FAPIRespAirportNotFound}},
)
async def ap_search(query: FAPIReqAPSearchQuery):
    ap = Airport.search(query)
    if ap.ap.valid:
        return {"status": "success", "airport": ap.ap.to_dict()}

    return construct_apnf_response("ap", Airport.suggest(ap.parse_result))


@app.get(
    "/route/info",
    response_model=FAPIRespRoute,
    responses={404: {"model": FAPIRespAirportNotFound}},
)
async def route_info(
    origin: FAPIReqAPSearchQuery,
    destination: FAPIReqAPSearchQuery,
):
    apsr0 = Airport.search(origin)
    if not apsr0.ap.valid:
        return construct_apnf_response("ap0", Airport.suggest(apsr0.parse_result))
    apsr1 = Airport.search(destination)
    if not apsr1.ap.valid:
        return construct_apnf_response("ap1", Airport.suggest(apsr1.parse_result))

    route = Route.create(apsr0.ap, apsr1.ap)
    return {
        "status": "success",
        "ap_origin": apsr0.ap.to_dict(),
        "ap_destination": apsr1.ap.to_dict(),
        "route": route.to_dict(),
    }


@app.get(
    "/ac_route/info",
    response_model=FAPIRespACRoute,
    response_model_exclude_unset=True,
    responses={404: {"model": FAPIRespAirportNotFound | FAPIRespAircraftNotFound}},
)
async def ac_route_info(
    origin: FAPIReqAPSearchQuery,
    destination: FAPIReqAPSearchQuery,
    ac: FAPIReqACSearchQuery,
    options: Annotated[FAPIReqACROptions, Depends()],
    user: Annotated[FAPIReqUser, Depends()],
):
    apsr0 = Airport.search(origin)
    if not apsr0.ap.valid:
        return construct_apnf_response("ap0", Airport.suggest(apsr0.parse_result))
    apsr1 = Airport.search(destination)
    if not apsr1.ap.valid:
        return construct_apnf_response("ap1", Airport.suggest(apsr1.parse_result))
    acsr = Aircraft.search(ac)
    if not acsr.ac.valid:
        return construct_acnf_response("ac", Aircraft.suggest(acsr.parse_result))

    ac_route = AircraftRoute.create(
        apsr0.ap,
        apsr1.ap,
        acsr.ac,
        options.to_core(acsr.ac.type),
        user.to_core(),
    )
    return {
        "status": "success",
        "ap_origin": apsr0.ap.to_dict(),
        "ap_destination": apsr1.ap.to_dict(),
        "ac": acsr.ac.to_dict(),
        "ac_route": ac_route.to_dict(),
    }


@app.get(
    "/ac_route/find",
    response_model=FAPIRespACRouteFind,
    responses={404: {"model": FAPIRespAirportNotFound | FAPIRespAircraftNotFound}},
)
async def ac_route_find_routes(
    ap0: FAPIReqAPSearchQuery,
    ac: FAPIReqACSearchQuery,
    options: Annotated[FAPIReqACROptions, Depends()],
    user: Annotated[FAPIReqUser, Depends()],
):
    apsr0 = Airport.search(ap0)
    if not apsr0.ap.valid:
        return construct_apnf_response("ap0", Airport.suggest(apsr0.parse_result))
    acsr = Aircraft.search(ac)
    if not acsr.ac.valid:
        return construct_acnf_response("ac", Aircraft.suggest(acsr.parse_result))

    rs = RoutesSearch(apsr0.ap, acsr.ac, options.to_core(acsr.ac.type), user.to_core())
    return ORJSONResponse(
        content={
            "status": "success",
            "destinations": [destination.to_dict() for destination in rs.get()],
        }
    )


server = Server(
    Config(
        app,
        host=cfg.api.HOST,
        port=cfg.api.PORT,
        reload=cfg.api.RELOAD,
        server_header=False,
        log_level=cfg.LOG_LEVEL.lower(),
    )
)


async def start(db_done: asyncio.Event):
    await db_done.wait()
    await server.serve()
