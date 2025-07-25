#include <iostream>
#include <string>
#include <algorithm>
#include <queue>

#include "include/db.hpp"
#include "include/ext/jaro.hpp"
#include "include/util.hpp"

shared_ptr<Database> Database::default_client = nullptr;
shared_ptr<Database> Database::Client(const string& home_dir) {
    if (!default_client) {
        default_client = make_shared<Database>();
        default_client->database = duckdb::make_uniq<DuckDB>(":memory:");
        default_client->connection = duckdb::make_uniq<Connection>(*default_client->database);
        std::cout << "duckdb: connected to " << home_dir << std::endl;

        CHECK_SUCCESS(default_client->connection->Query("SET home_directory = '" + home_dir + "';"));
    }
    return default_client;
}
shared_ptr<Database> Database::Client() {
    if (!default_client) {
        Database::Client(".");
    }
    return default_client;
}

void Database::populate_internal() {
    auto result = connection->Query("SELECT * FROM read_parquet('~/data/airports.parquet');");
    CHECK_SUCCESS_REF(result);
    idx_t i = 0;
    while (auto chunk = result->Fetch()) {
        for (idx_t j = 0; j < chunk->size(); j++, i++) {
            airports[i] = Airport(chunk, j);
        }
    }
    const uint16_t apid_breakpoints[] = {52,   178,  248,  318,  538,  542,  544,  552,  558,  562,  570,  572,  577,
                                         597,  1110, 1130, 1162, 1200, 1249, 1265, 1306, 1309, 1311, 1313, 1326, 1328,
                                         1356, 1358, 1378, 1381, 1388, 1391, 1468, 1481, 1513, 1528, 1532, 1537, 1540,
                                         1541, 1543, 1571, 1592, 1598, 1625, 1683, 1696, 2382, 2400, 2533, 2557, 2559,
                                         2566, 2573, 2577, 2591, 2597, 2610, 2627, 2630, 2646, 2648, 2656, 2660, 2662,
                                         2664, 2665, 2667, 2673, 3053, 3194, 3506, 3508, 3550, 3899, 3982};
    uint16_t start_bp = 1, offset = 1;
    for (uint16_t bp : apid_breakpoints) {
        for (uint16_t j = start_bp; j <= bp; j++) {
            airport_id_hashtable[j] = j - offset;
        }
        offset++;
        start_bp = bp + 1;
    }

    result = connection->Query("SELECT * FROM read_parquet('~/data/aircrafts.parquet');");
    CHECK_SUCCESS_REF(result);
    i = 0;
    while (auto chunk = result->Fetch()) {
        for (uint16_t j = 0; j < chunk->size(); j++, i++) {
            aircrafts[i] = Aircraft(chunk, j);
        }
    }

    result = connection->Query("SELECT yd, jd, fd, d FROM read_parquet('~/data/routes.parquet');");
    CHECK_SUCCESS_REF(result);
    i = 0;
    uint16_t x = 0, y = 0;
    while (auto chunk = result->Fetch()) {
        for (idx_t j = 0; j < chunk->size(); j++, i++) {
            pax_demands[i] = PaxDemand(
                chunk->GetValue(0, j).GetValue<uint16_t>(), chunk->GetValue(1, j).GetValue<uint16_t>(),
                chunk->GetValue(2, j).GetValue<uint16_t>()
            );
            y++;
            if (y == AIRPORT_COUNT) {
                x++;
                y = x + 1;
            }
            const double distance = chunk->GetValue(3, j).GetValue<double>();
            distances[x][y] = distance;
            distances[y][x] = distance;
        }
    }
}

const uint16_t missing_apids[] = {52,   178,  248,  318,  538,  542,  544,  552,  558,  562,  571,  572,  577,
                                  597,  1110, 1130, 1162, 1200, 1249, 1265, 1306, 1310, 1311, 1313, 1326, 1328,
                                  1356, 1358, 1378, 1381, 1388, 1391, 1468, 1481, 1513, 1528, 1532, 1537, 1540,
                                  1542, 1543, 1571, 1592, 1598, 1625, 1683, 1696, 2382, 2400, 2533, 2557, 2559,
                                  2566, 2573, 2577, 2591, 2597, 2610, 2627, 2630, 2647, 2648, 2656, 2660, 2662,
                                  2664, 2666, 2667, 2673, 3053, 3194, 3507, 3508, 3550, 3899};
Airport Database::get_airport_by_id(uint16_t id) {
    if (std::find(std::begin(missing_apids), std::end(missing_apids), id) != std::end(missing_apids)) return Airport();
    if (id > 3982) return Airport();
    return airports[airport_id_hashtable[id]];
}

// TODO: use unordered_map or gperf instead of linear search
Airport Database::get_airport_by_iata(const string& iata) {
    auto it = std::find_if(std::begin(airports), std::end(airports), [&](const Airport& a) { return a.iata == iata; });
    return it == std::end(airports) ? Airport() : *it;
}

Airport Database::get_airport_by_icao(const string& icao) {
    auto it = std::find_if(std::begin(airports), std::end(airports), [&](const Airport& a) { return a.icao == icao; });
    return it == std::end(airports) ? Airport() : *it;
}

Airport Database::get_airport_by_name(const string& name) {
    auto it = std::find_if(std::begin(airports), std::end(airports), [&](const Airport& a) {
        string db_name = a.name;
        std::transform(db_name.begin(), db_name.end(), db_name.begin(), ::toupper);
        return db_name == name;
    });
    return it == std::end(airports) ? Airport() : *it;
}

Airport Database::get_airport_by_fullname(const string& name) {
    auto it = std::find_if(std::begin(airports), std::end(airports), [&](const Airport& a) {
        string db_fullname = a.name + ", " + a.country;
        std::transform(db_fullname.begin(), db_fullname.end(), db_fullname.begin(), ::toupper);
        return db_fullname == name;
    });
    return it == std::end(airports) ? Airport() : *it;
}

Airport Database::get_airport_by_all(const string& all) {
    uint16_t id;
    if (str_to_uint16(all, id)) {
        Airport ap = get_airport_by_id(id);
        if (ap.valid) return ap;
    }
    auto it = std::find_if(std::begin(airports), std::end(airports), [&](const Airport& a) {
        string db_name = a.name;
        std::transform(db_name.begin(), db_name.end(), db_name.begin(), ::toupper);
        string db_fullname = db_name + ", " + a.country;
        std::transform(db_fullname.begin(), db_fullname.end(), db_fullname.begin(), ::toupper);
        return a.iata == all || a.icao == all || db_name == all || db_fullname == all;
    });
    return it == std::end(airports) ? Airport() : *it;
}

template <typename ScoreFn>
std::vector<Airport::Suggestion> Database::suggest_airport(const string& input, ScoreFn score_fn) {
    std::priority_queue<Airport::Suggestion, std::vector<Airport::Suggestion>, CompareSuggestion> pq;
    for (const Airport& ap : airports) {
        double score = score_fn(input, ap);
        if (pq.size() < 5) {
            pq.emplace(std::make_shared<Airport>(ap), score);
        } else if (score > pq.top().score) {
            pq.pop();
            pq.emplace(std::make_shared<Airport>(ap), score);
        }
    }
    std::vector<Airport::Suggestion> suggestions;
    while (!pq.empty()) {
        suggestions.insert(suggestions.begin(), pq.top());
        pq.pop();
    }
    return suggestions;
}

std::vector<Airport::Suggestion> Database::suggest_airport_by_iata(const string& iata) {
    return suggest_airport(iata, [](const string& input, const Airport& ap) {
        return jaro_winkler_distance(input, ap.iata);
    });
}

std::vector<Airport::Suggestion> Database::suggest_airport_by_icao(const string& icao) {
    return suggest_airport(icao, [](const string& input, const Airport& ap) {
        return jaro_winkler_distance(input, ap.icao);
    });
}

std::vector<Airport::Suggestion> Database::suggest_airport_by_name(const string& name) {
    return suggest_airport(name, [](const string& input, const Airport& ap) {
        string ap_name = ap.name;
        std::transform(ap_name.begin(), ap_name.end(), ap_name.begin(), ::toupper);
        return jaro_winkler_distance(input, ap_name);
    });
}

std::vector<Airport::Suggestion> Database::suggest_airport_by_fullname(const string& name) {
    return suggest_airport(name, [](const string& input, const Airport& ap) {
        string ap_fullname = ap.name + ", " + ap.country;
        std::transform(ap_fullname.begin(), ap_fullname.end(), ap_fullname.begin(), ::toupper);
        return jaro_winkler_distance(input, ap_fullname);
    });
}

std::vector<Airport::Suggestion> Database::suggest_airport_by_all(const string& name) {
    return suggest_airport(name, [](const string& input, const Airport& ap) {
        string ap_name = ap.name;
        std::transform(ap_name.begin(), ap_name.end(), ap_name.begin(), ::toupper);
        string ap_fullname = ap_name + ", " + ap.country;
        std::transform(ap_fullname.begin(), ap_fullname.end(), ap_fullname.begin(), ::toupper);
        return std::max(
            std::max(jaro_winkler_distance(input, ap.iata), jaro_winkler_distance(input, ap.icao)),
            std::max(jaro_winkler_distance(input, ap_name), jaro_winkler_distance(input, ap_fullname))
        );
    });
}

uint16_t Database::get_aircraft_idx_by_id(uint16_t id, uint8_t priority) {
    const std::map<uint16_t, uint16_t> idx_id_map{
        {1, 0},     {2, 4},     {3, 7},     {4, 8},     {5, 10},    {6, 12},    {7, 14},    {8, 17},    {9, 21},
        {10, 22},   {11, 23},   {12, 24},   {13, 25},   {14, 29},   {15, 30},   {16, 32},   {17, 34},   {18, 36},
        {19, 37},   {20, 38},   {21, 39},   {22, 41},   {23, 42},   {24, 43},   {25, 46},   {26, 47},   {27, 48},
        {28, 49},   {29, 50},   {30, 51},   {31, 52},   {32, 53},   {33, 57},   {34, 59},   {35, 60},   {36, 61},
        {37, 64},   {38, 66},   {39, 69},   {40, 72},   {41, 73},   {42, 74},   {43, 75},   {44, 76},   {45, 77},
        {46, 78},   {47, 79},   {48, 81},   {49, 82},   {50, 84},   {51, 85},   {52, 89},   {53, 93},   {55, 99},
        {56, 105},  {58, 106},  {59, 107},  {60, 108},  {61, 109},  {62, 112},  {63, 115},  {64, 118},  {66, 120},
        {67, 122},  {68, 124},  {69, 127},  {71, 129},  {72, 133},  {73, 137},  {74, 140},  {75, 143},  {76, 146},
        {85, 147},  {86, 148},  {87, 149},  {89, 151},  {90, 152},  {91, 153},  {92, 154},  {93, 155},  {94, 157},
        {95, 158},  {96, 159},  {97, 160},  {99, 161},  {100, 162}, {101, 163}, {102, 164}, {103, 166}, {104, 167},
        {105, 168}, {106, 169}, {107, 170}, {108, 171}, {109, 172}, {110, 173}, {111, 174}, {112, 175}, {113, 178},
        {114, 180}, {115, 182}, {116, 183}, {117, 184}, {118, 185}, {119, 186}, {120, 187}, {124, 190}, {126, 191},
        {127, 192}, {128, 193}, {129, 194}, {130, 195}, {131, 196}, {132, 197}, {133, 198}, {134, 199}, {135, 200},
        {136, 201}, {137, 202}, {138, 203}, {139, 204}, {140, 205}, {141, 206}, {142, 207}, {143, 208}, {144, 209},
        {145, 210}, {146, 211}, {147, 212}, {148, 213}, {149, 214}, {150, 216}, {151, 217}, {152, 218}, {153, 219},
        {154, 220}, {155, 221}, {156, 222}, {157, 223}, {158, 224}, {159, 225}, {160, 226}, {161, 227}, {162, 228},
        {163, 229}, {164, 230}, {165, 231}, {166, 232}, {167, 234}, {168, 235}, {169, 236}, {170, 238}, {171, 241},
        {172, 242}, {173, 243}, {177, 246}, {178, 247}, {179, 248}, {180, 249}, {181, 250}, {182, 251}, {183, 252},
        {184, 253}, {185, 254}, {186, 255}, {187, 256}, {189, 257}, {190, 258}, {191, 259}, {192, 260}, {193, 261},
        {194, 262}, {195, 264}, {196, 265}, {197, 266}, {198, 267}, {199, 268}, {200, 269}, {201, 270}, {202, 272},
        {203, 273}, {204, 274}, {205, 276}, {206, 277}, {207, 278}, {208, 279}, {209, 280}, {210, 281}, {211, 282},
        {212, 283}, {213, 284}, {214, 285}, {215, 286}, {216, 287}, {218, 288}, {219, 289}, {220, 290}, {221, 291},
        {222, 292}, {226, 293}, {227, 296}, {228, 299}, {229, 300}, {230, 303}, {231, 304}, {232, 305}, {233, 307},
        {234, 309}, {241, 310}, {242, 313}, {243, 319}, {244, 320}, {245, 322}, {246, 323}, {247, 324}, {248, 325},
        {249, 326}, {250, 328}, {251, 329}, {252, 330}, {253, 331}, {254, 332}, {255, 333}, {256, 334}, {257, 335},
        {258, 336}, {259, 337}, {260, 338}, {266, 339}, {267, 340}, {268, 341}, {269, 343}, {270, 344}, {271, 345},
        {272, 348}, {273, 349}, {274, 350}, {275, 351}, {276, 352}, {277, 354}, {281, 355}, {282, 356}, {283, 357},
        {284, 358}, {285, 359}, {287, 360}, {288, 362}, {289, 363}, {290, 364}, {291, 365}, {292, 366}, {293, 367},
        {294, 368}, {295, 369}, {298, 370}, {299, 371}, {300, 372}, {302, 374}, {303, 377}, {304, 379}, {305, 380},
        {306, 381}, {307, 383}, {308, 384}, {309, 385}, {310, 386}, {311, 390}, {312, 394}, {313, 395}, {314, 396},
        {315, 397}, {316, 399}, {317, 400}, {318, 401}, {320, 405}, {321, 407}, {322, 409}, {323, 413}, {324, 417},
        {325, 418}, {326, 419}, {327, 420}, {328, 421}, {329, 422}, {330, 423}, {331, 424}, {332, 425}, {333, 426},
        {334, 427}, {335, 428}, {336, 432}, {337, 434}, {338, 435}, {339, 436}, {340, 438}, {341, 439}, {342, 440},
        {343, 441}, {344, 442}, {345, 443}, {346, 444}, {347, 445}, {348, 446}, {349, 447}, {350, 448}, {351, 449},
        {352, 450}, {353, 451}, {355, 453}, {356, 454}, {357, 455}, {358, 457}, {359, 459}, {360, 460}, {361, 464},
        {362, 468}, {363, 471}, {364, 474}, {365, 475}, {366, 477}, {367, 479}, {368, 480}, {369, 482}, {370, 483},
        {371, 484}, {372, 485}, {373, 486}, {374, 487}, {375, 488}, {376, 489}, {377, 491}, {378, 492}, {379, 493},
        {380, 494}, {381, 495}, {382, 496}
    };
    auto search = idx_id_map.find(id);
    if (search != idx_id_map.end()) {
        auto next = std::next(search);
        if (next != idx_id_map.end()) {
            if (priority > next->second - search->second - 1) priority = 0;
        }
        return search->second + priority;
    }
    return 0;
}

const uint16_t missing_acids[] = {54,  57,  65,  70,  77,  78,  79,  80,  81,  82,  83,  84,  88,  98,  121, 122,
                                  123, 125, 174, 175, 176, 188, 217, 223, 224, 225, 235, 236, 237, 238, 239, 240,
                                  261, 262, 263, 264, 265, 278, 279, 280, 286, 296, 297, 301, 319, 354};
Aircraft Database::get_aircraft_by_id(uint16_t id, uint8_t priority) {
    if (std::find(std::begin(missing_acids), std::end(missing_acids), id) != std::end(missing_acids)) return Aircraft();
    if (id > 382) return Aircraft();
    return aircrafts[Database::get_aircraft_idx_by_id(id, priority)];
}

Aircraft Database::get_aircraft_by_shortname(const string& shortname, uint8_t priority) {
    auto it = std::find_if(std::begin(aircrafts), std::end(aircrafts), [&](const Aircraft& a) {
        return a.shortname == shortname && a.priority == priority;
    });
    return it == std::end(aircrafts) ? Aircraft() : *it;
}

Aircraft Database::get_aircraft_by_name(const string& name, uint8_t priority) {
    auto it = std::find_if(std::begin(aircrafts), std::end(aircrafts), [&](const Aircraft& a) {
        string db_name = a.name;
        std::transform(db_name.begin(), db_name.end(), db_name.begin(), ::tolower);
        return db_name == name && a.priority == priority;
    });
    return it == std::end(aircrafts) ? Aircraft() : *it;
}

Aircraft Database::get_aircraft_by_all(const string& shortname, uint8_t priority) {
    uint16_t id;
    if (str_to_uint16(shortname, id)) {
        Aircraft ac = get_aircraft_by_id(id, 0);
        if (ac.valid) return ac;
    }
    auto it = std::find_if(std::begin(aircrafts), std::end(aircrafts), [&](const Aircraft& a) {
        string db_name = a.name;
        std::transform(db_name.begin(), db_name.end(), db_name.begin(), ::tolower);
        return (a.shortname == shortname || db_name == shortname) && a.priority == priority;
    });
    return it == std::end(aircrafts) ? Aircraft() : *it;
}

template <typename ScoreFn>
std::vector<Aircraft::Suggestion> Database::suggest_aircraft(const string& input, ScoreFn score_fn) {
    std::priority_queue<Aircraft::Suggestion, std::vector<Aircraft::Suggestion>, CompareSuggestion> pq;
    for (const Aircraft& ac : aircrafts) {
        if (ac.priority != 0) continue;
        double score = score_fn(input, ac);
        if (pq.size() < 5) {
            pq.emplace(std::make_shared<Aircraft>(ac), score);
        } else if (score > pq.top().score) {
            pq.pop();
            pq.emplace(std::make_shared<Aircraft>(ac), score);
        }
    }
    std::vector<Aircraft::Suggestion> suggestions;
    while (!pq.empty()) {
        suggestions.insert(suggestions.begin(), pq.top());
        pq.pop();
    }
    return suggestions;
}

std::vector<Aircraft::Suggestion> Database::suggest_aircraft_by_shortname(const string& name) {
    return suggest_aircraft(name, [](const string& input, const Aircraft& ac) {
        return jaro_winkler_distance(input, ac.shortname);
    });
}

std::vector<Aircraft::Suggestion> Database::suggest_aircraft_by_name(const string& name) {
    return suggest_aircraft(name, [](const string& input, const Aircraft& ac) {
        string ac_name = ac.name;
        std::transform(ac_name.begin(), ac_name.end(), ac_name.begin(), ::tolower);
        return jaro_winkler_distance(input, ac_name);
    });
}

std::vector<Aircraft::Suggestion> Database::suggest_aircraft_by_all(const string& all) {
    return suggest_aircraft(all, [](const string& input, const Aircraft& ac) {
        string ac_name = ac.name;
        std::transform(ac_name.begin(), ac_name.end(), ac_name.begin(), ::tolower);
        return std::max(jaro_winkler_distance(input, ac.shortname), jaro_winkler_distance(input, ac_name));
    });
}

void init(string home_dir) {
    auto client = Database::Client(home_dir);
    client->populate_internal();
}

void _debug_query(string query) {
    auto client = Database::Client();

    auto result = client->connection->Query(query);
    result->Print();
}

#if BUILD_PYBIND == 1
#include "include/binder.hpp"
#include <optional>
#include <filesystem>

void pybind_init_db(py::module_& m) {
    py::module_ m_db = m.def_submodule("db");

    m_db.def(
            "init",
            [](std::optional<string> home_dir) {
                py::gil_scoped_acquire acquire;
                if (!home_dir.has_value()) {
                    string hdir = py::module::import("am4")
                                      .attr("__path__")
                                      .cast<py::list>()[0]
                                      .cast<string>();  // am4.__path__[0]
                    py::function urlretrieve = py::module::import("urllib.request").attr("urlretrieve");
                    if (!std::filesystem::exists(hdir + "/data")) {
                        std::cout << "WARN: data directory not found, creating..." << std::endl;
                        std::filesystem::create_directory(hdir + "/data");
                    }
                    for (const string fn : {"airports.parquet", "aircrafts.parquet", "routes.parquet"}) {
                        if (!std::filesystem::exists(hdir + "/data/" + fn)) {
                            std::cout << "WARN: " << fn << " not found, downloading from GitHub..." << std::endl;
                            urlretrieve(
                                "https://github.com/abc8747/am4/releases/latest/download/" + fn, hdir + "/data/" + fn
                            );
                        }
                    }
                    init(hdir);
                } else {
                    init(home_dir.value());
                }
                py::gil_scoped_release release;
            },
            "home_dir"_a = py::none()
    )
        .def("_debug_query", &_debug_query, "query"_a);

    py::module_ m_utils = m_db.def_submodule("utils");
    m_utils.def("jaro_distance", &jaro_distance, "a"_a, "b"_a)
        .def("jaro_winkler_distance", &jaro_winkler_distance, "a"_a, "b"_a);

    py::register_exception<DatabaseException>(m_db, "DatabaseException");
}
#endif