#include <math.h>
#include <vector>
#include <cmath>
#include <iostream>

#include "include/route.hpp"
#include "include/db.hpp"

using std::get;

Route::Route() : direct_distance(0.0), valid(false){};

// basic route meta
Route Route::create(const Airport& ap1, const Airport& ap2) {
    if (ap1.id == ap2.id) throw SameOdException();

    const auto& db = Database::Client();
    const uint16_t o_idx = db->airport_id_hashtable[ap1.id];
    const uint16_t d_idx = db->airport_id_hashtable[ap2.id];

    Route route;
    route.pax_demand = db->pax_demands[db->get_dbroute_idx(o_idx, d_idx)];
    route.direct_distance = db->distances[o_idx][d_idx];
    route.valid = true;
    return route;
}

inline double Route::calc_distance(double lat1, double lon1, double lat2, double lon2) {
    double dLat = (lat2 - lat1) * M_PI / 180.0;
    double dLon = (lon2 - lon1) * M_PI / 180.0;
    return 12742 *
           asin(
               sqrt(pow(sin(dLat / 2), 2) + cos(lat1 * M_PI / 180.0) * cos(lat2 * M_PI / 180.0) * pow(sin(dLon / 2), 2))
           );
}

inline double Route::calc_distance(const Airport& ap1, const Airport& ap2) {
    return calc_distance(ap1.lat, ap1.lng, ap2.lat, ap2.lng);
}

template <typename Cfg>
void tpd_sweep(
    const User& user,
    const AircraftRoute::Options& options,
    std::function<double()> est_max_tpd,
    std::function<Cfg(double)> calc_cfg,
    std::function<uint32_t(const Cfg&)> calc_max_income,
    const float ac_load,
    AircraftRoute* ar
) {
    // first, calculate the configuration for 1 aircraft
    double tpdpa = static_cast<double>(options.trips_per_day_per_ac);
    if (options.tpd_mode == AircraftRoute::Options::TPDMode::AUTO) {
        tpdpa = std::min(floor(24. / static_cast<double>(ar->flight_time)), floor(est_max_tpd()));
    }
    Cfg cfg = calc_cfg(tpdpa);
    if (options.tpd_mode != AircraftRoute::Options::TPDMode::AUTO && !cfg.valid) {
        ar->warnings.push_back(AircraftRoute::Warning::ERR_INSUFFICIENT_DEMAND);
        ar->valid = false;
        return;
    }
    // it's possible that the est_max_tpd() is still too high due to precision issues (though very unlikely)
    while (!cfg.valid) {
        tpdpa--;
        if (tpdpa <= 0) {
            ar->warnings.push_back(AircraftRoute::Warning::ERR_INSUFFICIENT_DEMAND);
            ar->valid = false;
            return;
        }
        cfg = calc_cfg(tpdpa);
    }

    double max_income = calc_max_income(cfg);
    if (options.tpd_mode == AircraftRoute::Options::TPDMode::STRICT) {
        ar->config = cfg;
        ar->max_income = max_income;
        ar->income = max_income * ac_load;
        ar->num_ac = 1;
        ar->trips_per_day_per_ac = tpdpa;
        ar->valid = true;
        uint16_t max_tpd_demand = static_cast<uint16_t>(est_max_tpd());
        max_tpd_demand -= max_tpd_demand % static_cast<uint16_t>(floor(24. / static_cast<double>(ar->flight_time)));
        ar->max_tpd = tpdpa == max_tpd_demand ? std::nullopt : std::optional<uint16_t>(max_tpd_demand);
        return;
    }
    /*
    Now, we progressively increase the aircraft per route - but if the total t/d is:
      - low: we can guarantee that the optimum class is filled first, but we end up wasting seats;
      - high: the max income is less optimal as seats are distributed, but this will allow better *total* income.
    for top-level gameplay, we often value route availability over profitability.

    When inflating, make sure that:
    - the demand is not exhausted;
    - the loss is greater than the max income bound;
    */
    double max_income_bnd = max_income * (1 - user.income_loss_tol);

    double num_ac = 1;
    for (double i_num_ac = num_ac + 1; i_num_ac < 200; i_num_ac++) {
        const auto i_cfg = calc_cfg(tpdpa * i_num_ac);
        if (!i_cfg.valid) break;

        const auto i_max_income = calc_max_income(i_cfg);
        if (i_max_income < max_income_bnd) break;

        cfg = i_cfg;
        max_income = i_max_income;
        num_ac = i_num_ac;
    }
    ar->config = cfg;
    ar->max_income = max_income;
    ar->income = max_income * ac_load;
    ar->num_ac = num_ac;
    ar->trips_per_day_per_ac = tpdpa;
    ar->valid = true;
    ar->max_tpd = std::nullopt;
}

// TODO: use one template function for both pax and cargo
template <bool is_vip>
inline void AircraftRoute::update_pax_details(
    uint16_t ac_capacity, const AircraftRoute::Options& options, const User& user
) {
    const Aircraft::PaxConfig::Algorithm config_algorithm =
        std::holds_alternative<std::monostate>(options.config_algorithm)
            ? Aircraft::PaxConfig::Algorithm::AUTO
            : get<Aircraft::PaxConfig::Algorithm>(options.config_algorithm);
    const PaxDemand load_adj_pd = this->route.pax_demand / user.load;
    auto est_max_tpd = [&]() -> double {
        return static_cast<double>(load_adj_pd.y + load_adj_pd.j * 2 + load_adj_pd.f * 3) /
               static_cast<double>(ac_capacity);
    };
    auto calc_cfg = [&](double tpd) {
        return Aircraft::PaxConfig::calc_pax_conf(
            load_adj_pd / tpd, ac_capacity, this->route.direct_distance, user.game_mode, config_algorithm
        );
    };

    const auto tkt = [&]() {
        if constexpr (is_vip)
            return VIPTicket::from_optimal(this->route.direct_distance, user.game_mode);
        else
            return PaxTicket::from_optimal(this->route.direct_distance, user.game_mode);
    }();
    auto calc_max_income = [&](const Aircraft::PaxConfig& cfg) -> uint32_t {
        return (cfg.y * tkt.y + cfg.j * tkt.j + cfg.f * tkt.f);
    };
    tpd_sweep<Aircraft::PaxConfig>(user, options, est_max_tpd, calc_cfg, calc_max_income, user.load, this);
    this->ticket = tkt;
}

inline void AircraftRoute::update_cargo_details(
    uint32_t ac_capacity, const AircraftRoute::Options& options, const User& user
) {
    const Aircraft::CargoConfig::Algorithm config_algorithm =
        std::holds_alternative<std::monostate>(options.config_algorithm)
            ? Aircraft::CargoConfig::Algorithm::AUTO
            : get<Aircraft::CargoConfig::Algorithm>(options.config_algorithm);
    const CargoDemand load_adj_cd = CargoDemand(this->route.pax_demand);

    auto est_max_tpd = [&]() -> double {
        double k_h = 1. + static_cast<double>(user.h_training) / 100;
        double k_l = 1. + static_cast<double>(user.l_training) / 100;
        return (
            ((k_h / k_l / 0.7) * static_cast<double>(load_adj_cd.l) +
             static_cast<double>(load_adj_cd.h) / (k_h * static_cast<double>(ac_capacity)))
        );
    };
    auto calc_cfg = [&](double trips_per_day) {
        return Aircraft::CargoConfig::calc_cargo_conf(
            load_adj_cd / user.cargo_load / trips_per_day, ac_capacity, user.l_training, user.h_training,
            config_algorithm
        );
    };
    const CargoTicket tkt = CargoTicket::from_optimal(this->route.direct_distance, user.game_mode);
    auto calc_income = [&](const Aircraft::CargoConfig& cfg) -> double {
        return ((1 + user.l_training / 100.0) * cfg.l * 0.7 * tkt.l + (1 + user.h_training / 100.0) * cfg.h * tkt.h) *
               ac_capacity / 100.0;
    };
    tpd_sweep<Aircraft::CargoConfig>(user, options, est_max_tpd, calc_cfg, calc_income, user.cargo_load, this);
    this->ticket = tkt;
}

AircraftRoute::AircraftRoute() : valid(false){};
AircraftRoute::Options::Options(
    TPDMode tpd_mode,
    uint16_t trips_per_day_per_ac,
    double max_distance,
    double min_distance,
    float max_flight_time,
    float min_flight_time,
    ConfigAlgorithm config_algorithm,
    SortBy sort_by
)
    : tpd_mode(tpd_mode),
      trips_per_day_per_ac(trips_per_day_per_ac),
      max_distance(max_distance),
      min_distance(min_distance),
      max_flight_time(max_flight_time),
      min_flight_time(min_flight_time),
      config_algorithm(config_algorithm),
      sort_by(sort_by) {
    if (tpd_mode == AircraftRoute::Options::TPDMode::AUTO && trips_per_day_per_ac != 1)
        std::cerr << "WARN: trips_per_day_per_ac is ignored when tpd_mode is AUTO" << std::endl;
};
AircraftRoute AircraftRoute::create(
    const Airport& a0, const Airport& a1, const Aircraft& ac, const AircraftRoute::Options& options, const User& user
) {
    AircraftRoute acr;
    acr.route = Route::create(a0, a1);
    acr._ac_type = ac.type;
    acr.max_tpd = std::nullopt;

    if (user.game_mode == User::GameMode::REALISM && (a1.rwy < ac.rwy)) {
        // NOTE: departing from runways that are too short is actually allowed
        // for realism player. just exercise caution not to make routes the other way
        acr.warnings.push_back(AircraftRoute::Warning::ERR_RWY_TOO_SHORT);
        return acr;
    }
    if (acr.route.direct_distance > options.max_distance) {
        acr.warnings.push_back(AircraftRoute::Warning::ERR_DISTANCE_ABOVE_SPECIFIED);
        return acr;
    } else if (acr.route.direct_distance < options.min_distance) {
        acr.warnings.push_back(AircraftRoute::Warning::ERR_DISTANCE_BELOW_SPECIFIED);
        return acr;
    } else if (acr.route.direct_distance > 2 * ac.range) {
        acr.warnings.push_back(AircraftRoute::Warning::ERR_DISTANCE_TOO_LONG);
        return acr;
    } else if (acr.route.direct_distance < 100) {
        acr.warnings.push_back(AircraftRoute::Warning::ERR_DISTANCE_TOO_SHORT);
        return acr;
    } else if (acr.route.direct_distance < 1000) {
        acr.warnings.push_back(AircraftRoute::Warning::REDUCED_CONTRIBUTION);
    }
    acr.needs_stopover = acr.route.direct_distance > ac.range;
    acr.stopover = acr.needs_stopover ? Stopover::find_by_efficiency(a0, a1, ac, user.game_mode) : Stopover();
    if (acr.needs_stopover && !acr.stopover.exists) {
        acr.warnings.push_back(AircraftRoute::Warning::ERR_NO_STOPOVER);
        return acr;
    }
    const double full_distance = acr.stopover.exists ? acr.stopover.full_distance : acr.route.direct_distance;
    acr.flight_time =
        static_cast<float>(full_distance) / (ac.speed * (user.game_mode == User::GameMode::EASY ? 1.5f : 1.0f));
    if (acr.flight_time > options.max_flight_time) {
        acr.warnings.push_back(AircraftRoute::Warning::ERR_FLIGHT_TIME_ABOVE_SPECIFIED);
        return acr;
    } else if (acr.flight_time < options.min_flight_time) {
        acr.warnings.push_back(AircraftRoute::Warning::ERR_FLIGHT_TIME_BELOW_SPECIFIED);
        return acr;
    }
    if (options.tpd_mode != Options::TPDMode::AUTO &&
        acr.flight_time > 24.0f / static_cast<float>(options.trips_per_day_per_ac)) {
        acr.warnings.push_back(AircraftRoute::Warning::ERR_TRIPS_PER_DAY_TOO_HIGH);
        return acr;
    }
    switch (ac.type) {
        case Aircraft::Type::PAX: {
            acr.update_pax_details<false>(static_cast<uint16_t>(ac.capacity), options, user);
            if (!acr.valid) return acr;
            acr.co2 = AircraftRoute::calc_co2(ac, get<Aircraft::PaxConfig>(acr.config), full_distance, user);
            break;
        }
        case Aircraft::Type::CARGO: {
            acr.update_cargo_details(static_cast<uint32_t>(ac.capacity), options, user);
            if (!acr.valid) return acr;
            acr.co2 = AircraftRoute::calc_co2(ac, get<Aircraft::CargoConfig>(acr.config), full_distance, user);
            break;
        }
        case Aircraft::Type::VIP: {
            acr.update_pax_details<true>(static_cast<uint16_t>(ac.capacity), options, user);
            if (!acr.valid) return acr;
            acr.co2 = AircraftRoute::calc_co2(ac, get<Aircraft::PaxConfig>(acr.config), full_distance, user);
            break;
        }
    }
    acr.fuel = AircraftRoute::calc_fuel(ac, full_distance, user);
    acr.acheck_cost = static_cast<float>(ac.check_cost * (user.game_mode == User::GameMode::EASY ? 0.5 : 1.0)) *
                      ceil(acr.flight_time * (user.game_mode == User::GameMode::EASY ? 1.5 : 1.0)) /
                      static_cast<float>(ac.maint);
    acr.repair_cost =
        ac.cost / 1000.0 * 0.0075 *
        (1 - 2 * user.repair_training / 100.0);  // each flight adds random [0, 1.5]% wear, each tp decreases wear by 2%
    acr.profit =
        (acr.income - acr.fuel * user.fuel_price / 1000.0 - acr.co2 * user.co2_price / 1000.0 - acr.acheck_cost -
         acr.repair_cost);
    acr.ci = 200;
    acr.contribution = AircraftRoute::calc_contribution(full_distance, user, 200);

    acr.valid = true;
    return acr;
}

AircraftRoute::Stopover::Stopover() : exists(false) {}
AircraftRoute::Stopover::Stopover(const Airport& airport, double full_distance)
    : airport(airport), full_distance(full_distance), exists(true) {}
AircraftRoute::Stopover AircraftRoute::Stopover::find_by_efficiency(
    const Airport& origin, const Airport& destination, const Aircraft& aircraft, User::GameMode game_mode
) {
    const auto& db = Database::Client();
    const auto& airports = db->airports;
    const auto& distances = db->distances;
    Airport candidate = Airport();
    double candidate_distance = 99999;

    const double ac_range = static_cast<double>(aircraft.range);
    const uint16_t o_idx = db->airport_id_hashtable[origin.id];
    const uint16_t d_idx = db->airport_id_hashtable[destination.id];
    // d_o & d_d will catch cases where idx == o_idx || idx == d_idx
    if (game_mode == User::GameMode::EASY) {
        for (uint16_t idx = 0; idx < AIRPORT_COUNT; idx++) {
            const Airport& ap = airports[idx];
            const double d_o = distances[o_idx][idx];
            if (d_o > ac_range || d_o < 100.0) continue;
            const double d_d = distances[d_idx][idx];
            if (d_d > ac_range || d_d < 100.0) continue;
            if (d_o + d_d < candidate_distance) {
                candidate = ap;
                candidate_distance = d_o + d_d;
            }
        }
    } else {
        const uint16_t rwy_requirement = aircraft.rwy;
        for (uint16_t idx = 0; idx < AIRPORT_COUNT; idx++) {
            const Airport& ap = airports[idx];
            if (ap.rwy < rwy_requirement) continue;
            const double d_o = distances[o_idx][idx];
            if (d_o > ac_range || d_o < 100.0) continue;
            const double d_d = distances[d_idx][idx];
            if (d_d > ac_range || d_d < 100.0) continue;
            if (d_o + d_d < candidate_distance) {
                candidate = ap;
                candidate_distance = d_o + d_d;
            }
        }
    }

    if (!candidate.valid) return Stopover();
    return Stopover(candidate, candidate_distance);
}

const string AircraftRoute::Stopover::repr(const Stopover& stopover) {
    if (!stopover.exists) return "<Stopover NONEXISTENT>";
    return "<Stopover airport=" + Airport::repr(stopover.airport) +
           " full_distance=" + to_string(stopover.full_distance) + ">";
}

inline double AircraftRoute::estimate_load(double reputation, double autoprice_ratio, bool has_stopover) {
    if (autoprice_ratio >
        1) {  // normal (sorta triangular?) distribution, [Z+(0: .00019, 1: .0068, 2: .0092), max: .001] * reputation
        if (has_stopover) {
            return 0.0085855 * reputation;
        } else {
            return 0.0090435 * reputation;
        }
    } else {  // uniform distribution: +- 0.00052 * reputation
        double base_load;
        if (has_stopover) {
            base_load = 0.0090312 * reputation;
        } else {
            base_load = 0.0095265 * reputation;
        }
        return (base_load - 1) * autoprice_ratio + 1;
    }
}

inline double AircraftRoute::calc_fuel(const Aircraft& ac, double distance, const User& user, uint8_t ci) {
    return (
        (1 - user.fuel_training / 100.0) * ceil(distance * 100.0) / 100.0 * ac.fuel *  // (ac.fuel_mod ? 0.9 : 1)
        (ci / 500.0 + 0.6)
    );
}

inline double AircraftRoute::calc_co2(
    const Aircraft& ac, const Aircraft::PaxConfig& cfg, double distance, const User& user, uint8_t ci
) {
    double ac_load = ac.type == Aircraft::Type::CARGO ? user.cargo_load : user.load;
    return (
        (1 - user.co2_training / 100.0) *
        (ceil(distance * 100.0) / 100.0 * ac.co2 * ((cfg.y + cfg.j * 2 + cfg.f * 3) * ac_load) + (cfg.y + cfg.j + cfg.f)
        ) *
        (ci / 2000.0 + 0.9)
    );
}

inline double AircraftRoute::calc_co2(
    const Aircraft& ac, const Aircraft::CargoConfig& cfg, double distance, const User& user, uint8_t ci
) {
    double ac_load = ac.type == Aircraft::Type::CARGO ? user.cargo_load : user.load;
    return (
        (1 - user.co2_training / 100.0) *
        (ceil(distance * 100.0) / 100.0 * ac.co2 *
             ((cfg.l / 100.0 * 0.7 / 1000.0 + cfg.h / 100.0 / 500.0) * ac_load * ac.capacity) +
         ((cfg.l / 100.0 * 0.7 + cfg.h / 100.0) * ac.capacity)) *
        (ci / 2000.0 + 0.9)
    );
}

inline float AircraftRoute::calc_contribution(double distance, const User& user, uint8_t ci) {
    float multiplier = 0.0064f;
    if (distance > 10000)
        multiplier = 0.0048f;
    else if (distance > 6000)
        multiplier = 0.0032f;

    float contribution = std::min(multiplier * static_cast<float>(distance) * (3 - ci / 100.0f), 152.0f);
    if (user.game_mode == User::GameMode::REALISM) contribution *= 1.5f;
    return contribution;
}

const string Route::repr(const Route& r) {
    string s;
    if (r.valid) {
        s = "<Route direct_distance=" + to_string(r.direct_distance) + " dem=" + to_string(r.pax_demand.y) + "|" +
            to_string(r.pax_demand.j) + "|" + to_string(r.pax_demand.f) + ">";
    } else {
        s = "<Route INVALID>";
    }
    return s;
}

inline const string to_string(const AircraftRoute::Warning& warning) {
    switch (warning) {
        case AircraftRoute::Warning::ERR_RWY_TOO_SHORT:
            return "Runway too short";
        case AircraftRoute::Warning::ERR_DISTANCE_ABOVE_SPECIFIED:
            return "Distance above specified limit";
        case AircraftRoute::Warning::ERR_DISTANCE_BELOW_SPECIFIED:
            return "Distance below specified limit";
        case AircraftRoute::Warning::ERR_DISTANCE_TOO_LONG:
            return "Distance too long";
        case AircraftRoute::Warning::ERR_DISTANCE_TOO_SHORT:
            return "Distance too short (<100km)";
        case AircraftRoute::Warning::REDUCED_CONTRIBUTION:
            return "Reduced contribution (<1000km)";
        case AircraftRoute::Warning::ERR_NO_STOPOVER:
            return "No stopovers are possible";
        case AircraftRoute::Warning::ERR_FLIGHT_TIME_ABOVE_SPECIFIED:
            return "Flight time above specified limit";
        case AircraftRoute::Warning::ERR_FLIGHT_TIME_BELOW_SPECIFIED:
            return "Flight time below specified limit";
        case AircraftRoute::Warning::ERR_INSUFFICIENT_DEMAND:
            return "Insufficient demand";
        case AircraftRoute::Warning::ERR_TRIPS_PER_DAY_TOO_HIGH:
            return "Trips per day per aircraft is too high for the flight time";
        default:
            return "Unknown error";
    }
}

const string AircraftRoute::repr(const AircraftRoute& ar) {
    string s;
    if (ar.valid) {
        s = "<AircraftRoute route=" + Route::repr(ar.route);
        switch (ar._ac_type) {
            case Aircraft::Type::VIP:
            case Aircraft::Type::PAX:
                s += " config=" + Aircraft::PaxConfig::repr(get<Aircraft::PaxConfig>(ar.config));
                break;
            case Aircraft::Type::CARGO:
                s += " config=" + Aircraft::CargoConfig::repr(get<Aircraft::CargoConfig>(ar.config));
                break;
        }
        switch (ar._ac_type) {
            case Aircraft::Type::VIP:
                s += " ticket=" + VIPTicket::repr(get<VIPTicket>(ar.ticket));
                break;
            case Aircraft::Type::PAX:
                s += " ticket=" + PaxTicket::repr(get<PaxTicket>(ar.ticket));
                break;
            case Aircraft::Type::CARGO:
                s += " ticket.cargo_ticket=" + CargoTicket::repr(get<CargoTicket>(ar.ticket));
                break;
        }
        s += " max_income=" + to_string(ar.max_income) + " income=" + to_string(ar.income);
        s += " fuel=" + to_string(ar.fuel) + " co2=" + to_string(ar.co2);
        s += " needs_stopover=" + to_string(ar.needs_stopover) +
             " stopover=" + AircraftRoute::Stopover::repr(ar.stopover);
        s += " warnings=[";
        for (const AircraftRoute::Warning& warning : ar.warnings) {
            s += to_string(warning) + ",";
        }
        s += "]>";
    } else {
        s = "<AircraftRoute INVALID>";
    }
    return s;
}

Destination::Destination(const Airport& origin, const Airport& destination, const AircraftRoute& route)
    : origin(origin), airport(destination), ac_route(route) {}

std::vector<Destination> RoutesSearch::get() const {
    std::vector<Destination> destinations;
    const auto& db = Database::Client();

    const uint16_t rwy_requirement = this->user.game_mode == User::GameMode::EASY ? 0 : this->aircraft.rwy;
    for (const Airport& origin : this->origins) {
        for (const Airport& ap : db->airports) {
            if (ap.rwy < rwy_requirement || ap.id == origin.id) continue;
            const AircraftRoute ar = AircraftRoute::create(origin, ap, this->aircraft, this->options, this->user);
            if (!ar.valid) continue;
            destinations.emplace_back(origin, ap, ar);
        }
    }
    auto cmp = this->options.sort_by == AircraftRoute::Options::SortBy::PER_TRIP
                   ? [](const Destination& a, const Destination& b) { return a.ac_route.profit > b.ac_route.profit; }
                   : [](const Destination& a, const Destination& b) {
                         return a.ac_route.profit * a.ac_route.trips_per_day_per_ac >
                                b.ac_route.profit * b.ac_route.trips_per_day_per_ac;
                     };
    std::sort(destinations.begin(), destinations.end(), cmp);
    return destinations;
}

#if BUILD_PYBIND == 1
#include "include/binder.hpp"

py::dict to_dict(const Route& r) {
    return py::dict(
        "pax_demand"_a = to_dict(r.pax_demand), "cargo_demand"_a = to_dict(CargoDemand(r.pax_demand)),
        "direct_distance"_a = r.direct_distance
    );
}

py::dict to_dict(const AircraftRoute::Stopover& s) {
    return s.exists ? py::dict("airport"_a = to_dict(s.airport), "full_distance"_a = s.full_distance, "exists"_a = true)
                    : py::dict("exists"_a = false);
}

py::list to_list(const vector<AircraftRoute::Warning>& warnings) {
    py::list l;
    for (const AircraftRoute::Warning& w : warnings) {
        l.append(to_string(w));
    }
    return l;
}

py::dict to_dict(const AircraftRoute& ar) {
    py::dict d(
        "route"_a = to_dict(ar.route), "warnings"_a = to_list(ar.warnings), "valid"_a = false, "max_tpd"_a = ar.max_tpd
    );

    if (std::any_of(std::begin(ar.warnings), std::end(ar.warnings), [](const AircraftRoute::Warning& w) {
            return w == AircraftRoute::Warning::ERR_RWY_TOO_SHORT ||
                   w == AircraftRoute::Warning::ERR_DISTANCE_ABOVE_SPECIFIED ||
                   w == AircraftRoute::Warning::ERR_DISTANCE_TOO_LONG ||
                   w == AircraftRoute::Warning::ERR_DISTANCE_TOO_SHORT;
        }))
        return d;

    d["needs_stopover"] = ar.needs_stopover;
    d["stopover"] = to_dict(ar.stopover);

    if (std::any_of(std::begin(ar.warnings), std::end(ar.warnings), [](const AircraftRoute::Warning& w) {
            return w == AircraftRoute::Warning::ERR_NO_STOPOVER;
        }))
        return d;

    d["flight_time"] = ar.flight_time;

    if (std::any_of(std::begin(ar.warnings), std::end(ar.warnings), [](const AircraftRoute::Warning& w) {
            return w == AircraftRoute::Warning::ERR_FLIGHT_TIME_ABOVE_SPECIFIED ||
                   w == AircraftRoute::Warning::ERR_INSUFFICIENT_DEMAND;
        }))
        return d;

    d["trips_per_day_per_ac"] = ar.trips_per_day_per_ac;
    d["num_ac"] = ar.num_ac;

    switch (ar._ac_type) {
        case Aircraft::Type::PAX:
            d["config"] = to_dict(get<Aircraft::PaxConfig>(ar.config));
            d["ticket"] = to_dict(get<PaxTicket>(ar.ticket));
            break;
        case Aircraft::Type::VIP:
            d["config"] = to_dict(get<Aircraft::PaxConfig>(ar.config));
            d["ticket"] = to_dict(get<VIPTicket>(ar.ticket));
            break;
        case Aircraft::Type::CARGO:
            d["config"] = to_dict(get<Aircraft::CargoConfig>(ar.config));
            d["ticket"] = to_dict(get<CargoTicket>(ar.ticket));
            break;
    }
    d["max_income"] = ar.max_income;
    d["income"] = ar.income;
    d["fuel"] = ar.fuel;
    d["co2"] = ar.co2;
    d["acheck_cost"] = ar.acheck_cost;
    d["repair_cost"] = ar.repair_cost;
    d["profit"] = ar.profit;
    d["ci"] = ar.ci;
    d["contribution"] = ar.contribution;
    d["valid"] = ar.valid;

    return d;
}

py::dict to_dict(const Destination& d, bool include_origin) {
    py::dict dest;
    if (include_origin) {
        // not including everything to save bandwidth
        dest["origin"] = py::dict("id"_a = d.origin.id, "iata"_a = d.origin.iata, "icao"_a = d.origin.icao);
    }
    dest["airport"] = to_dict(d.airport);
    dest["ac_route"] = to_dict(d.ac_route);
    return dest;
}

std::map<string, py::list> _get_columns(const RoutesSearch& rs, const vector<Destination>& dests, bool include_origin) {
    // for use in csv generation via pyarrow.Table.from_pydict & downstream statistical analysis
    // assuming dests to be all valid
    std::map<string, py::list> cols;
    for (const Destination& dest : dests) {
        if (include_origin) {
            cols["00|orig.id"].append(dest.origin.id);
            cols["03|orig.iata"].append(dest.origin.iata);
            cols["04|orig.icao"].append(dest.origin.icao);
        }
        cols["10|dest.id"].append(dest.airport.id);
        cols["11|dest.name"].append(dest.airport.name);
        cols["12|dest.country"].append(dest.airport.country);
        cols["13|dest.iata"].append(dest.airport.iata);
        cols["14|dest.icao"].append(dest.airport.icao);
        // for use in map display
        cols["98|dest.lat"].append(dest.airport.lat);
        cols["99|dest.lng"].append(dest.airport.lng);
        auto& acr = dest.ac_route;
        if (dest.ac_route.stopover.exists) {
            cols["15|stop.id"].append(acr.stopover.airport.id);
            cols["16|stop.name"].append(acr.stopover.airport.name);
            cols["17|stop.country"].append(acr.stopover.airport.country);
            cols["18|stop.iata"].append(acr.stopover.airport.iata);
            cols["19|stop.icao"].append(acr.stopover.airport.icao);
            cols["20|full_dist"].append(acr.stopover.full_distance);
        } else {
            cols["15|stop.id"].append(py::none());
            cols["16|stop.name"].append(py::none());
            cols["17|stop.country"].append(py::none());
            cols["18|stop.iata"].append(py::none());
            cols["19|stop.icao"].append(py::none());
            cols["20|full_dist"].append(py::none());
        }
        if (rs.aircraft.type == Aircraft::Type::CARGO) {
            auto dem = CargoDemand(acr.route.pax_demand);
            auto& cfg = get<Aircraft::CargoConfig>(acr.config);
            auto& tkt = get<CargoTicket>(acr.ticket);
            cols["21|dem.l"].append(dem.l);
            cols["22|dem.h"].append(dem.h);
            cols["24|cfg.l"].append(cfg.l);
            cols["25|cfg.h"].append(cfg.h);
            cols["27|tkt.l"].append(tkt.l);
            cols["28|tkt.h"].append(tkt.h);
        } else {
            auto& dem = acr.route.pax_demand;
            auto& cfg = get<Aircraft::PaxConfig>(acr.config);
            auto& tkt =
                rs.aircraft.type == Aircraft::Type::VIP ? get<VIPTicket>(acr.ticket) : get<PaxTicket>(acr.ticket);
            cols["21|dem.y"].append(dem.y);
            cols["22|dem.j"].append(dem.j);
            cols["23|dem.f"].append(dem.f);
            cols["24|cfg.y"].append(cfg.y);
            cols["25|cfg.j"].append(cfg.j);
            cols["26|cfg.f"].append(cfg.f);
            cols["27|tkt.y"].append(tkt.y);
            cols["28|tkt.j"].append(tkt.j);
            cols["29|tkt.f"].append(tkt.f);
        }
        cols["30|direct_dist"].append(acr.route.direct_distance);
        cols["31|time"].append(acr.flight_time);
        cols["32|trips_pd_pa"].append(acr.trips_per_day_per_ac);
        cols["33|num_ac"].append(acr.num_ac);
        cols["34|income"].append(acr.income);
        cols["35|fuel"].append(acr.fuel);
        cols["36|co2"].append(acr.co2);
        cols["37|chk_cost"].append(acr.acheck_cost);
        cols["38|repair_cost"].append(acr.repair_cost);
        cols["39|profit_pt"].append(acr.profit);
        cols["40|ci"].append(acr.ci);
        cols["41|contrib_pt"].append(acr.contribution);
    }
    return cols;
}

void pybind_init_route(py::module_& m) {
    py::module_ m_route = m.def_submodule("route");

    py::class_<AircraftRoute> acr_class(m_route, "AircraftRoute");

    py::register_exception<SameOdException>(m_route, "SameOdException");

    py::class_<Route>(m_route, "Route")
        .def_readonly("pax_demand", &Route::pax_demand)
        .def_readonly("direct_distance", &Route::direct_distance)
        .def_readonly("valid", &Route::valid)
        .def_static("create", &Route::create, "ap0"_a, "ap1"_a)
        .def_static(
            "calc_distance", py::overload_cast<double, double, double, double>(&Route::calc_distance), "lat1"_a,
            "lon1"_a, "lat2"_a, "lon2"_a
        )
        .def_static(
            "calc_distance", py::overload_cast<const Airport&, const Airport&>(&Route::calc_distance), "a0"_a, "a1"_a
        )
        .def("__repr__", &Route::repr)
        .def("to_dict", py::overload_cast<const Route&>(&to_dict));

    py::class_<AircraftRoute::Options> acr_options_class(acr_class, "Options");
    py::enum_<AircraftRoute::Options::TPDMode>(acr_options_class, "TPDMode")
        .value("AUTO", AircraftRoute::Options::TPDMode::AUTO)
        .value("STRICT_ALLOW_MULTIPLE_AC", AircraftRoute::Options::TPDMode::STRICT_ALLOW_MULTIPLE_AC)
        .value("STRICT", AircraftRoute::Options::TPDMode::STRICT);
    py::enum_<AircraftRoute::Options::SortBy>(acr_options_class, "SortBy")
        .value("PER_TRIP", AircraftRoute::Options::SortBy::PER_TRIP)
        .value("PER_AC_PER_DAY", AircraftRoute::Options::SortBy::PER_AC_PER_DAY);
    acr_options_class
        .def(
            py::init<
                AircraftRoute::Options::TPDMode, uint16_t, double, double, float, float,
                AircraftRoute::Options::ConfigAlgorithm, AircraftRoute::Options::SortBy>(),
            py::arg_v("tpd_mode", AircraftRoute::Options::TPDMode::AUTO, "TPDMode.AUTO"), "trips_per_day_per_ac"_a = 1,
            "max_distance"_a = MAX_DISTANCE, "min_distance"_a = 0.0, "max_flight_time"_a = 24.0f,
            "min_flight_time"_a = 0.0f, "config_algorithm"_a = std::monostate(),
            py::arg_v("sort_by", AircraftRoute::Options::SortBy::PER_TRIP, "SortBy.PER_TRIP")
        )
        .def_readwrite("tpd_mode", &AircraftRoute::Options::tpd_mode)
        .def_readwrite("trips_per_day_per_ac", &AircraftRoute::Options::trips_per_day_per_ac)
        .def_readwrite("max_distance", &AircraftRoute::Options::max_distance)
        .def_readwrite("min_distance", &AircraftRoute::Options::min_distance)
        .def_readwrite("max_flight_time", &AircraftRoute::Options::max_flight_time)
        .def_readwrite("min_flight_time", &AircraftRoute::Options::min_flight_time)
        .def_readwrite("config_algorithm", &AircraftRoute::Options::config_algorithm)
        .def_readwrite("sort_by", &AircraftRoute::Options::sort_by);

    py::class_<AircraftRoute::Stopover>(acr_class, "Stopover")
        .def_readonly("airport", &AircraftRoute::Stopover::airport)
        .def_readonly("full_distance", &AircraftRoute::Stopover::full_distance)
        .def_readonly("exists", &AircraftRoute::Stopover::exists)
        .def_static(
            "find_by_efficiency", &AircraftRoute::Stopover::find_by_efficiency, "origin"_a, "destination"_a,
            "aircraft"_a, "game_mode"_a
        )
        .def("__repr__", &AircraftRoute::Stopover::repr)
        .def("to_dict", py::overload_cast<const AircraftRoute::Stopover&>(&to_dict));

    py::enum_<AircraftRoute::Warning>(acr_class, "Warning")
        .value("ERR_RWY_TOO_SHORT", AircraftRoute::Warning::ERR_RWY_TOO_SHORT)
        .value("ERR_DISTANCE_ABOVE_SPECIFIED", AircraftRoute::Warning::ERR_DISTANCE_ABOVE_SPECIFIED)
        .value("ERR_DISTANCE_BELOW_SPECIFIED", AircraftRoute::Warning::ERR_DISTANCE_BELOW_SPECIFIED)
        .value("ERR_DISTANCE_TOO_LONG", AircraftRoute::Warning::ERR_DISTANCE_TOO_LONG)
        .value("ERR_DISTANCE_TOO_SHORT", AircraftRoute::Warning::ERR_DISTANCE_TOO_SHORT)
        .value("REDUCED_CONTRIBUTION", AircraftRoute::Warning::REDUCED_CONTRIBUTION)
        .value("ERR_NO_STOPOVER", AircraftRoute::Warning::ERR_NO_STOPOVER)
        .value("ERR_FLIGHT_TIME_ABOVE_SPECIFIED", AircraftRoute::Warning::ERR_FLIGHT_TIME_ABOVE_SPECIFIED)
        .value("ERR_FLIGHT_TIME_BELOW_SPECIFIED", AircraftRoute::Warning::ERR_FLIGHT_TIME_BELOW_SPECIFIED)
        .value("ERR_INSUFFICIENT_DEMAND", AircraftRoute::Warning::ERR_INSUFFICIENT_DEMAND)
        .value("ERR_TRIPS_PER_DAY_TOO_HIGH", AircraftRoute::Warning::ERR_TRIPS_PER_DAY_TOO_HIGH)
        .def("to_str", py::overload_cast<const AircraftRoute::Warning&>(&to_string));

    acr_class.def_readonly("route", &AircraftRoute::route)
        .def_readonly("config", &AircraftRoute::config)
        .def_readonly("ticket", &AircraftRoute::ticket)
        .def_readonly("trips_per_day_per_ac", &AircraftRoute::trips_per_day_per_ac)
        .def_readonly("max_income", &AircraftRoute::max_income)
        .def_readonly("income", &AircraftRoute::income)
        .def_readonly("fuel", &AircraftRoute::fuel)
        .def_readonly("co2", &AircraftRoute::co2)
        .def_readonly("acheck_cost", &AircraftRoute::acheck_cost)
        .def_readonly("repair_cost", &AircraftRoute::repair_cost)
        .def_readonly("profit", &AircraftRoute::profit)
        .def_readonly("flight_time", &AircraftRoute::flight_time)
        .def_readonly("num_ac", &AircraftRoute::num_ac)
        .def_readonly("ci", &AircraftRoute::ci)
        .def_readonly("contribution", &AircraftRoute::contribution)
        .def_readonly("needs_stopover", &AircraftRoute::needs_stopover)
        .def_readonly("stopover", &AircraftRoute::stopover)
        .def_readonly("warnings", &AircraftRoute::warnings)
        .def_readonly("valid", &AircraftRoute::valid)
        .def_readonly("max_tpd", &AircraftRoute::max_tpd)
        .def_static(
            "create", &AircraftRoute::create, "ap0"_a, "ap1"_a, "ac"_a,
            py::arg_v("options", AircraftRoute::Options(), "AircraftRoute.Options()"),
            py::arg_v("user", User::Default(), "am4.utils.game.User.Default()")
        )
        .def_static(
            "estimate_load", &AircraftRoute::estimate_load, "reputation"_a = 87, "autoprice_ratio"_a = 1.06,
            "has_stopover"_a = false
        )
        .def_static(
            "calc_fuel", &AircraftRoute::calc_fuel, "ac"_a, "distance"_a,
            py::arg_v("user", User::Default(), "am4.utils.game.User.Default()"), "ci"_a = 200
        )
        .def_static(
            "calc_co2",
            py::overload_cast<const Aircraft&, const Aircraft::PaxConfig&, double, const User&, uint8_t>(
                &AircraftRoute::calc_co2
            ),
            "ac"_a, "cfg"_a, "distance"_a, py::arg_v("user", User::Default(), "am4.utils.game.User.Default()"),
            "ci"_a = 200
        )
        .def_static(
            "calc_co2",
            py::overload_cast<const Aircraft&, const Aircraft::CargoConfig&, double, const User&, uint8_t>(
                &AircraftRoute::calc_co2
            ),
            "ac"_a, "cfg"_a, "distance"_a, py::arg_v("user", User::Default(), "am4.utils.game.User.Default()"),
            "ci"_a = 200
        )
        .def("__repr__", &AircraftRoute::repr)
        .def("to_dict", py::overload_cast<const AircraftRoute&>(&to_dict));

    py::class_<Destination>(m_route, "Destination")
        .def_readonly("origin", &Destination::origin)
        .def_readonly("airport", &Destination::airport)
        .def_readonly("ac_route", &Destination::ac_route)
        .def("to_dict", py::overload_cast<const Destination&, const bool>(&to_dict), "include_origin"_a = false);

    py::class_<RoutesSearch>(m_route, "RoutesSearch")
        .def(
            py::init<const vector<Airport>&, const Aircraft&, const AircraftRoute::Options&, const User&>(), "ap0"_a,
            "ac"_a, py::arg_v("options", AircraftRoute::Options(), "AircraftRoute.Options()"),
            py::arg_v("user", User::Default(), "am4.utils.game.User.Default()")
        )
        .def("get", &RoutesSearch::get, py::call_guard<py::gil_scoped_release>())
        .def(
            "_get_columns",
            py::overload_cast<const RoutesSearch&, const vector<Destination>&, const bool>(&_get_columns), "dests"_a,
            "include_origin"_a = false
        );
}
#endif