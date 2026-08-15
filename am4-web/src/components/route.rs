use crate::components::airport::APSearch;
use crate::components::format_flight_time_csv;
use crate::components::format_flight_time_hms;
use crate::components::format_thousands;
use crate::components::icons::DownloadIcon;
use crate::db::Data;
use am4::aircraft::custom::CustomAircraft;
use am4::aircraft::AircraftType;
use am4::airport::Airport;
use am4::route::config::ConfigAlgorithm;
use am4::route::demand::{CargoDemand, PaxDemand};
use am4::route::metrics::ConfigVariant;
use am4::route::search::ferry::FerryRoutes;
use am4::route::search::schedule::{
    CiStrategy, NumAircraftStrategy, ScheduleStrategy, SearchConfig, SortBy, TripsPerDay,
    TripsPerDayStrategy,
};
use am4::route::search::AbstractRoutes;
use am4::route::ticket::Ticket;
use am4::route::{Ci, Distance, FlightTime};
use am4::user::{AirportCodePref, CsvTimeFormat, GameMode, Settings};
use am4::utils::Filter;
use leptos::html::Select;
use leptos::prelude::*;
use leptos::wasm_bindgen::JsCast;
use leptos::web_sys::js_sys;
use std::fmt::Write;
use std::num::NonZeroU8;
use std::str::FromStr;
use web_sys::{Blob, HtmlAnchorElement, Url};

#[derive(Clone, Default, PartialEq, Copy)]
pub struct RouteStats {
    pub count: usize,
    pub time_ms: f64,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum SearchMode {
    #[default]
    AnyDestination,
    SpecificDestination,
    Sell,
}

#[derive(Clone, PartialEq)]
pub struct WebScheduledRoute {
    pub origin: Airport,
    pub destination: Airport,
    pub stopover: Option<Airport>,
    pub direct_distance: Distance,
    pub total_distance: Distance,
    pub flight_time: FlightTime,
    pub ci: Ci,
    pub contribution: f32,
    pub trips_per_day: u8,
    pub num_aircraft: u8,
    pub config: ConfigVariant,
    pub ticket: Ticket,
    pub revenue: f32,
    pub fuel: f32,
    pub co2: f32,
    pub acheck_cost: f32,
    pub repair_cost: f32,
    pub profit: f32,
    pub demand: PaxDemand,
}

#[derive(Clone, PartialEq)]
pub struct WebFerryRoute {
    pub origin: Airport,
    pub destination: Airport,
    pub direct_distance: Distance,
    pub flight_time: FlightTime,
    pub fuel: f32,
    pub sale_price: f32,
    pub profit: f32,
}

#[derive(Clone, PartialEq)]
pub enum WebRoute {
    Scheduled(WebScheduledRoute),
    Ferry(WebFerryRoute),
}

impl WebRoute {
    fn profit(&self) -> f32 {
        match self {
            Self::Scheduled(r) => r.profit,
            Self::Ferry(r) => r.profit,
        }
    }

    fn trips_per_day(&self) -> u8 {
        match self {
            Self::Scheduled(r) => r.trips_per_day,
            Self::Ferry(_) => 1,
        }
    }

    fn profit_per_day(&self) -> f32 {
        self.profit() * self.trips_per_day() as f32
    }
}

fn trigger_file_download(csv_content: &str, filename: &str) {
    let blob = Blob::new_with_str_sequence(&js_sys::Array::of1(&csv_content.into())).expect("blob");
    let url = Url::create_object_url_with_blob(&blob).expect("url");
    let window = web_sys::window().expect("window");
    let doc = window.document().expect("document");
    let a = doc
        .create_element("a")
        .expect("element")
        .dyn_into::<HtmlAnchorElement>()
        .expect("anchor");

    a.set_href(&url);
    a.set_download(filename);
    a.click();
    Url::revoke_object_url(&url).ok();
}

fn download_csv(routes: Vec<WebRoute>, csv_time_format: CsvTimeFormat) {
    if routes.is_empty() {
        return;
    }

    match &routes[0] {
        WebRoute::Scheduled(_) => {
            let scheduled = routes
                .into_iter()
                .filter_map(|r| match r {
                    WebRoute::Scheduled(s) => Some(s),
                    WebRoute::Ferry(_) => None,
                })
                .collect();
            download_scheduled_csv(scheduled, csv_time_format);
        }
        WebRoute::Ferry(_) => {
            let ferry = routes
                .into_iter()
                .filter_map(|r| match r {
                    WebRoute::Scheduled(_) => None,
                    WebRoute::Ferry(f) => Some(f),
                })
                .collect();
            download_ferry_csv(ferry, csv_time_format);
        }
    }
}

fn download_scheduled_csv(routes: Vec<WebScheduledRoute>, csv_time_format: CsvTimeFormat) {
    if routes.is_empty() {
        return;
    }

    let is_cargo = matches!(routes[0].config, ConfigVariant::Cargo(_));
    let mut csv = String::with_capacity(routes.len() * 200);

    csv.push_str(
        "orig.id,orig.iata,orig.icao,\
        dest.id,dest.name,dest.country,dest.iata,dest.icao,dest.lat,dest.lng,\
        stop.id,stop.name,stop.country,stop.iata,stop.icao,stop.full_dist,",
    );

    if is_cargo {
        csv.push_str("dem.l,dem.h,cfg.l,cfg.h,tkt.l,tkt.h,");
    } else {
        csv.push_str("dem.y,dem.j,dem.f,cfg.y,cfg.j,cfg.f,tkt.y,tkt.j,tkt.f,");
    }

    csv.push_str(
        "direct_dist,time,trips_pd_pa,num_ac,income,fuel,co2,chk_cost,repair_cost,profit_pt,ci,contrib_pt\n",
    );

    let escape = |s: &str| -> String {
        if s.contains(',') || s.contains('"') || s.contains('\n') {
            format!("\"{}\"", s.replace('"', "\"\""))
        } else {
            s.to_string()
        }
    };

    for r in routes {
        let _ = write!(csv, "{},{},{},", r.origin.id, r.origin.iata, r.origin.icao);
        let _ = write!(
            csv,
            "{},{},{},{},{},{:.6},{:.6},",
            r.destination.id,
            escape(&r.destination.name.to_string()),
            escape(&r.destination.country),
            r.destination.iata,
            r.destination.icao,
            r.destination.location.lat,
            r.destination.location.lng
        );

        if let Some(stop) = r.stopover {
            let _ = write!(
                csv,
                "{},{},{},{},{},{:.2},",
                stop.id,
                escape(&stop.name.to_string()),
                escape(&stop.country),
                stop.iata,
                stop.icao,
                r.total_distance.get()
            );
        } else {
            csv.push_str(",,,,,,");
        }

        if is_cargo {
            let dem: CargoDemand = (&r.demand).into();
            let cfg = match r.config {
                ConfigVariant::Cargo(c) => c,
                _ => unreachable!(),
            };
            let tkt = match r.ticket {
                Ticket::Cargo(t) => t,
                _ => unreachable!(),
            };
            let _ = write!(
                csv,
                "{},{},{},{},{:.2},{:.2},",
                dem.l, dem.h, cfg.l, cfg.h, tkt.l, tkt.h
            );
        } else {
            let dem = r.demand;
            let cfg = match r.config {
                ConfigVariant::Pax(c) => c,
                _ => unreachable!(),
            };
            let (ty, tj, tf) = match r.ticket {
                Ticket::Pax(t) => (t.y, t.j, t.f),
                Ticket::VIP(t) => (t.y, t.j, t.f),
                _ => unreachable!(),
            };
            let _ = write!(
                csv,
                "{},{},{},{},{},{},{},{},{},",
                dem.y, dem.j, dem.f, cfg.y, cfg.j, cfg.f, ty, tj, tf
            );
        }

        let _ = writeln!(
            csv,
            "{:.2},{},{},{},{:.0},{:.2},{:.2},{:.2},{:.2},{:.0},{},{:.2}",
            r.direct_distance.get(),
            format_flight_time_csv(r.flight_time, csv_time_format),
            r.trips_per_day,
            r.num_aircraft,
            r.revenue,
            r.fuel,
            r.co2,
            r.acheck_cost,
            r.repair_cost,
            r.profit,
            r.ci.get(),
            r.contribution
        );
    }

    trigger_file_download(&csv, "routes.csv");
}

fn download_ferry_csv(routes: Vec<WebFerryRoute>, csv_time_format: CsvTimeFormat) {
    if routes.is_empty() {
        return;
    }

    let mut csv = String::with_capacity(routes.len() * 120);
    csv.push_str(
        "orig.id,orig.iata,orig.icao,dest.id,dest.name,dest.country,dest.iata,dest.icao,direct_dist,time,fuel,sale_price,profit\n",
    );

    let escape = |s: &str| -> String {
        if s.contains(',') || s.contains('"') || s.contains('\n') {
            format!("\"{}\"", s.replace('"', "\"\""))
        } else {
            s.to_string()
        }
    };

    for r in routes {
        let _ = writeln!(
            csv,
            "{},{},{},{},{},{},{},{},{:.2},{},{:.2},{:.2},{:.2}",
            r.origin.id,
            r.origin.iata,
            r.origin.icao,
            r.destination.id,
            escape(&r.destination.name.to_string()),
            escape(&r.destination.country),
            r.destination.iata,
            r.destination.icao,
            r.direct_distance.get(),
            format_flight_time_csv(r.flight_time, csv_time_format),
            r.fuel,
            r.sale_price,
            r.profit
        );
    }

    trigger_file_download(&csv, "ferry-routes.csv");
}

#[component]
fn FilterInput(
    label: &'static str,
    min_val: RwSignal<String>,
    max_val: RwSignal<String>,
    error: ReadSignal<Option<String>>,
) -> impl IntoView {
    view! {
        <label>
            {label} <div class="range-inputs">
                <input
                    type="text"
                    placeholder="Min"
                    prop:value=min_val
                    class:invalid=move || error.get().is_some()
                    on:input=move |ev| min_val.set(event_target_value(&ev))
                />
                <span class="sep">"-"</span>
                <input
                    type="text"
                    placeholder="Max"
                    prop:value=max_val
                    class:invalid=move || error.get().is_some()
                    on:input=move |ev| max_val.set(event_target_value(&ev))
                />
            </div> {move || { error.get().map(|e| view! { <span class="input-error">{e}</span> }) }}
        </label>
    }
}

#[component]
fn StrategyInput(
    label: &'static str,
    value: RwSignal<String>,
    is_active: RwSignal<bool>,
    active_label: &'static str,
    error: ReadSignal<Option<String>>,
    #[prop(optional)] disable_button: Option<Signal<bool>>,
) -> impl IntoView {
    view! {
        <label>
            {label} <div class="input-with-btn">
                <input
                    type="number"
                    prop:value=value
                    class:invalid=move || error.get().is_some()
                    on:input=move |ev| {
                        let v = event_target_value(&ev);
                        value.set(v.clone());
                        if v.trim().is_empty() {
                            is_active.set(true);
                        } else {
                            is_active.set(false);
                        }
                    }
                />
                <button
                    class:active=move || is_active.get() && disable_button.is_none_or(|s| !s.get())
                    class:disabled=move || disable_button.is_some_and(|s| s.get())
                    prop:disabled=move || disable_button.is_some_and(|s| s.get())
                    on:click=move |_| {
                        if disable_button.is_none_or(|s| !s.get()) {
                            is_active.set(true);
                            value.set(String::new());
                        }
                    }
                >
                    {active_label}
                </button>
            </div> {move || { error.get().map(|e| view! { <span class="input-error">{e}</span> }) }}
        </label>
    }
}

#[component]
fn ConfigAlgoInput(
    algo: RwSignal<ConfigAlgorithm>,
    ac_type: Memo<Option<AircraftType>>,
) -> impl IntoView {
    let select_ref = NodeRef::<Select>::new();

    Effect::new(move |_| {
        let a = algo.get();
        if let Some(el) = select_ref.get() {
            el.set_value(&a.to_string());
        }
    });

    view! {
        <label class:disabled=move || {
            ac_type.get().is_none()
        }>
            "Config Algorithm"
            <select
                node_ref=select_ref
                class:disabled=move || ac_type.get().is_none()
                prop:disabled=move || ac_type.get().is_none()
                on:change=move |ev| {
                    let val = event_target_value(&ev);
                    if let Ok(alg) = val.parse() {
                        algo.set(alg);
                    }
                }
            >
                <option value="auto">"Auto"</option>
                {move || {
                    ac_type
                        .get()
                        .map(|t| match t {
                            AircraftType::Pax | AircraftType::Vip => {
                                view! {
                                    <option value="fjy">"F>J>Y"</option>
                                    <option value="fyj">"F>Y>J"</option>
                                    <option value="jfy">"J>F>Y"</option>
                                    <option value="jyf">"J>Y>F"</option>
                                    <option value="yfj">"Y>F>J"</option>
                                    <option value="yjf">"Y>J>F"</option>
                                    <option value="y">"Y Only"</option>
                                    <option value="j">"J Only"</option>
                                    <option value="f">"F Only"</option>
                                }
                                    .into_any()
                            }
                            AircraftType::Cargo => {
                                view! {
                                    <option value="lh">"L>H"</option>
                                    <option value="hl">"H>L"</option>
                                    <option value="l">"L Only"</option>
                                    <option value="h">"H Only"</option>
                                }
                                    .into_any()
                            }
                        })
                }}
                <option value="spread">"Spread"</option>
            </select>
        </label>
    }
}

#[derive(Default)]
struct InputErrors {
    dist: Option<String>,
    ft: Option<String>,
    num_ac: Option<String>,
    tpd: Option<String>,
    ci: Option<String>,
}

struct ParsedSearchInputs {
    distance_filter: Filter<Distance>,
    flight_time_filter: Filter<FlightTime>,
    schedule: ScheduleStrategy,
    ci: CiStrategy,
    sort_by: SortBy,
    inflate_distance_with_stopover: bool,
    config: ConfigAlgorithm,
}

fn parse_range_values(
    min: &str,
    max: &str,
    parser: &dyn Fn(&str) -> Result<f32, String>,
) -> Result<(Option<f32>, Option<f32>), String> {
    let parse_val = |s: &str| -> Result<Option<f32>, String> {
        if s.trim().is_empty() {
            return Ok(None);
        }
        parser(s).map(Some)
    };

    let min_res = parse_val(min)?;
    let max_res = parse_val(max)?;

    if let (Some(min), Some(max)) = (min_res, max_res) {
        if min < 0. || max < 0. {
            return Err("Must be positive".into());
        }
        if min > max {
            return Err("Min > Max".into());
        }
    }

    Ok((min_res, max_res))
}

#[allow(clippy::too_many_arguments)]
fn parse_search_inputs(
    mode: SearchMode,
    d_min: &str,
    d_max: &str,
    f_min: &str,
    f_max: &str,
    n_ac: &str,
    n_ac_is_max: bool,
    tpd: &str,
    tpd_is_max: bool,
    ci_s: &str,
    ci_is_auto: bool,
    stopover: bool,
    sort_by: SortBy,
    config: ConfigAlgorithm,
    show_distance_filters: bool,
    show_num_ac: bool,
    show_tpd: bool,
    show_ci: bool,
) -> (Option<ParsedSearchInputs>, InputErrors) {
    let mut errors = InputErrors::default();

    let dist_parser = |s: &str| s.parse::<f32>().map_err(|_| "Invalid number".to_string());
    let ft_parser = |s: &str| {
        FlightTime::from_str(s)
            .map(|ft| ft.get())
            .map_err(|_| "Invalid format".to_string())
    };

    let distance_filter = if show_distance_filters {
        match parse_range_values(d_min, d_max, &dist_parser) {
            Ok((l, r)) => match (l, r) {
                (Some(min), Some(max)) => {
                    Filter::Range(Distance::new_unchecked(min)..Distance::new_unchecked(max))
                }
                (Some(min), None) => Filter::RangeFrom(Distance::new_unchecked(min)..),
                (None, Some(max)) => Filter::RangeTo(..Distance::new_unchecked(max)),
                (None, None) => Filter::RangeFull,
            },
            Err(e) => {
                errors.dist = Some(e);
                Filter::RangeFull
            }
        }
    } else {
        Filter::RangeFull
    };

    let ft_max_specified = show_distance_filters && !f_max.trim().is_empty();
    let flight_time_filter = if show_distance_filters {
        match parse_range_values(f_min, f_max, &ft_parser) {
            Ok((l, r)) => match (l, r) {
                (Some(min), Some(max)) => {
                    Filter::Range(FlightTime::new_unchecked(min)..FlightTime::new_unchecked(max))
                }
                (Some(min), None) => Filter::RangeFrom(FlightTime::new_unchecked(min)..),
                (None, Some(max)) => Filter::RangeTo(..FlightTime::new_unchecked(max)),
                (None, None) => Filter::RangeFull,
            },
            Err(e) => {
                errors.ft = Some(e);
                Filter::RangeFull
            }
        }
    } else {
        Filter::RangeFull
    };

    let num_aircraft = if show_num_ac {
        if n_ac_is_max {
            NumAircraftStrategy::Maximise
        } else if n_ac.trim().is_empty() {
            errors.num_ac = Some("Required".into());
            NumAircraftStrategy::Strict(NonZeroU8::new(1).unwrap())
        } else {
            match n_ac.parse::<u8>().ok().and_then(NonZeroU8::new) {
                Some(n) => NumAircraftStrategy::Strict(n),
                None => {
                    errors.num_ac = Some("Must be > 0".into());
                    NumAircraftStrategy::Strict(NonZeroU8::new(1).unwrap())
                }
            }
        }
    } else {
        NumAircraftStrategy::Strict(NonZeroU8::new(1).unwrap())
    };

    let trips_per_day = if show_tpd {
        if tpd_is_max {
            TripsPerDayStrategy::Maximise
        } else if tpd.trim().is_empty() {
            errors.tpd = Some("Required".into());
            TripsPerDayStrategy::Strict(TripsPerDay::new(1).unwrap())
        } else {
            match tpd.parse::<u8>().ok().and_then(TripsPerDay::new) {
                Some(t) => TripsPerDayStrategy::Strict(t),
                None => {
                    errors.tpd = Some("Must be > 0".into());
                    TripsPerDayStrategy::Strict(TripsPerDay::new(1).unwrap())
                }
            }
        }
    } else {
        TripsPerDayStrategy::Strict(TripsPerDay::new(1).unwrap())
    };

    let ci = if show_ci {
        if ci_is_auto && ft_max_specified {
            CiStrategy::AlignConstraint
        } else if ci_s.trim().is_empty() {
            errors.ci = Some("Required".into());
            CiStrategy::Strict(Ci::MAX)
        } else {
            match ci_s.parse::<u8>().ok().and_then(|v| Ci::new(v).ok()) {
                Some(c) => CiStrategy::Strict(c),
                None => {
                    errors.ci = Some("0-200".into());
                    CiStrategy::Strict(Ci::MAX)
                }
            }
        }
    } else {
        CiStrategy::Strict(Ci::MAX)
    };

    if errors.dist.is_some()
        || errors.ft.is_some()
        || errors.num_ac.is_some()
        || errors.tpd.is_some()
        || errors.ci.is_some()
    {
        return (None, errors);
    }

    (
        Some(ParsedSearchInputs {
            distance_filter,
            flight_time_filter,
            schedule: ScheduleStrategy {
                trips_per_day,
                num_aircraft,
            },
            ci,
            sort_by,
            inflate_distance_with_stopover: mode == SearchMode::AnyDestination && stopover,
            config,
        }),
        errors,
    )
}

fn execute_sell_mode(
    data: &Data,
    ap_sel: &[Airport],
    custom_ac: &am4::aircraft::Aircraft,
    user_settings: &Settings,
    gm: &GameMode,
    parsed: &ParsedSearchInputs,
) -> Vec<WebRoute> {
    let mut all_results = Vec::new();

    for origin in ap_sel {
        let abstract_routes = AbstractRoutes::new(
            &data.airports,
            &data.distances,
            origin,
            data.airports.data(),
        );
        let concrete = abstract_routes.with_aircraft(custom_ac, gm);
        let ferry_routes =
            FerryRoutes::new(concrete.routes().to_vec(), custom_ac, user_settings, *gm);

        for r in ferry_routes.routes() {
            if !parsed.distance_filter.contains(&r.direct_distance)
                || !parsed.flight_time_filter.contains(&r.flight_time)
            {
                continue;
            }

            all_results.push(WebRoute::Ferry(WebFerryRoute {
                origin: origin.clone(),
                destination: r.destination.clone(),
                direct_distance: r.direct_distance,
                flight_time: r.flight_time,
                fuel: r.fuel,
                sale_price: r.sale_price,
                profit: r.profit,
            }));
        }
    }

    all_results.sort_by(|a, b| b.profit().partial_cmp(&a.profit()).unwrap());
    all_results
}

fn execute_scheduled_mode(
    data: &Data,
    ap_sel: &[Airport],
    ap_dest: &[Airport],
    mode: SearchMode,
    custom_ac: &am4::aircraft::Aircraft,
    user_settings: &Settings,
    gm: &GameMode,
    parsed: &ParsedSearchInputs,
) -> Option<Vec<WebRoute>> {
    let demands = data.demands.as_ref()?;
    let search_config = SearchConfig {
        user_settings,
        distance_filter: parsed.distance_filter.clone(),
        flight_time_filter: parsed.flight_time_filter.clone(),
        schedule: parsed.schedule.clone(),
        ci: parsed.ci.clone(),
        sort_by: parsed.sort_by.clone(),
        inflate_distance_with_stopover: parsed.inflate_distance_with_stopover,
        config: parsed.config,
    };

    let mut all_results = Vec::new();

    for origin in ap_sel {
        let destinations = if mode == SearchMode::SpecificDestination {
            ap_dest
        } else {
            data.airports.data()
        };

        let abstract_routes =
            AbstractRoutes::new(&data.airports, &data.distances, origin, destinations);
        let concrete = abstract_routes.with_aircraft(custom_ac, gm);
        let scheduled = concrete.schedule(demands, &data.distances, &search_config);

        for r in scheduled.routes() {
            all_results.push(WebRoute::Scheduled(WebScheduledRoute {
                origin: origin.clone(),
                destination: r.destination.clone(),
                stopover: r.stopover.as_ref().map(|s| s.0.clone()),
                direct_distance: r.direct_distance,
                total_distance: r.total_distance,
                flight_time: r.flight_time,
                ci: r.ci,
                contribution: r.contribution,
                trips_per_day: r.trips_per_day.get(),
                num_aircraft: r.num_aircraft.get(),
                config: r.config,
                ticket: r.ticket,
                revenue: r.revenue,
                fuel: r.fuel,
                co2: r.co2,
                acheck_cost: r.acheck_cost,
                repair_cost: r.repair_cost,
                profit: r.profit,
                demand: demands[(origin.idx, r.destination.idx)],
            }));
        }
    }

    if mode == SearchMode::AnyDestination {
        match parsed.sort_by {
            SortBy::ProfitPerTrip => {
                all_results.sort_by(|a, b| b.profit().partial_cmp(&a.profit()).unwrap());
            }
            SortBy::ProfitPerAcPerDay => {
                all_results
                    .sort_by(|a, b| b.profit_per_day().partial_cmp(&a.profit_per_day()).unwrap());
            }
        }
    }

    Some(all_results)
}

#[component]
pub fn RouteOptions(
    #[prop(into)] ac_selected: RwSignal<Vec<CustomAircraft>>,
    #[prop(into)] ap_selected: RwSignal<Vec<Airport>>,
    #[prop(into)] ap_destination: RwSignal<Vec<Airport>>,
    #[prop(into)] ap_destination_active: RwSignal<Option<Airport>>,
    #[prop(into)] search_mode: RwSignal<SearchMode>,
    #[prop(into)] set_routes: WriteSignal<Vec<WebRoute>>,
    #[prop(into)] set_stats: WriteSignal<RouteStats>,
) -> impl IntoView {
    let database = expect_context::<StoredValue<Option<Data>>>();
    let settings = expect_context::<RwSignal<Settings>>();
    let game_mode = expect_context::<RwSignal<GameMode>>();

    let dist_min = RwSignal::new(String::new());
    let dist_max = RwSignal::new(String::new());
    let (dist_error, set_dist_error) = signal(None::<String>);

    let ft_min = RwSignal::new(String::new());
    let ft_max = RwSignal::new(String::new());
    let (ft_error, set_ft_error) = signal(None::<String>);

    let sort_by = RwSignal::new(SortBy::ProfitPerAcPerDay);

    let num_ac_input = RwSignal::new(String::from("1"));
    let num_ac_max = RwSignal::new(false);
    let (num_ac_error, set_num_ac_error) = signal(None::<String>);

    let tpd_input = RwSignal::new(String::new());
    let tpd_max = RwSignal::new(true);
    let (tpd_error, set_tpd_error) = signal(None::<String>);

    let ci_input = RwSignal::new(String::from("200"));
    let ci_auto = RwSignal::new(false);
    let (ci_error, set_ci_error) = signal(None::<String>);

    let stopover_mode = RwSignal::new(false);
    let config_algo = RwSignal::new(ConfigAlgorithm::Auto);

    let ac_type = Memo::new(move |_| ac_selected.get().first().map(|c| c.aircraft.r#type.clone()));
    let show_destination_input =
        Memo::new(move |_| search_mode.get() == SearchMode::SpecificDestination);
    let show_distance_filters =
        Memo::new(move |_| search_mode.get() != SearchMode::SpecificDestination);
    let show_num_ac = Memo::new(move |_| search_mode.get() == SearchMode::AnyDestination);
    let show_tpd = Memo::new(move |_| search_mode.get() != SearchMode::Sell);
    let show_ci = Memo::new(move |_| search_mode.get() != SearchMode::Sell);
    let can_execute = Memo::new(move |_| {
        !ac_selected.get().is_empty()
            && !ap_selected.get().is_empty()
            && (!show_destination_input.get() || !ap_destination.get().is_empty())
    });
    let route_settings = Memo::new(move |_| {
        settings.with(|s| {
            (
                s.training.wear,
                s.training.repair,
                s.training.l,
                s.training.h,
                s.training.fuel,
                s.training.co2,
                s.fuel_price,
                s.co2_price,
                s.load,
                s.cargo_load,
                s.revenue_loss_tol,
                s.allow_invalid_tpd,
            )
        })
    });

    Effect::new(move |_| {
        let max = dist_max.get();
        if max.trim().is_empty() && stopover_mode.get_untracked() {
            stopover_mode.set(false);
        }
    });

    Effect::new(move |_| {
        let ac_sel = ac_selected.get();
        let ap_sel = ap_selected.get();
        let ap_dest = ap_destination.get();
        let mode = search_mode.get();
        route_settings.track();
        let user_settings = settings.get_untracked();
        let gm = game_mode.get();
        let sort = sort_by.get();
        let d_min = dist_min.get();
        let d_max = dist_max.get();
        let f_min = ft_min.get();
        let f_max = ft_max.get();
        let n_ac = num_ac_input.get();
        let n_ac_is_max = num_ac_max.get();
        let tpd = tpd_input.get();
        let tpd_is_max = tpd_max.get();
        let ci_s = ci_input.get();
        let ci_is_auto = ci_auto.get();
        let stopover = stopover_mode.get();
        let algo = config_algo.get();

        if !can_execute.get() {
            set_routes.set(vec![]);
            set_stats.set(RouteStats::default());
            set_dist_error.set(None);
            set_ft_error.set(None);
            set_num_ac_error.set(None);
            set_tpd_error.set(None);
            set_ci_error.set(None);
            return;
        }

        let custom_ac = ac_sel[0].effective();

        let (parsed, errors) = parse_search_inputs(
            mode,
            &d_min,
            &d_max,
            &f_min,
            &f_max,
            &n_ac,
            n_ac_is_max,
            &tpd,
            tpd_is_max,
            &ci_s,
            ci_is_auto,
            stopover,
            sort,
            algo,
            show_distance_filters.get(),
            show_num_ac.get(),
            show_tpd.get(),
            show_ci.get(),
        );

        set_dist_error.set(errors.dist);
        set_ft_error.set(errors.ft);
        set_num_ac_error.set(errors.num_ac);
        set_tpd_error.set(errors.tpd);
        set_ci_error.set(errors.ci);

        let Some(parsed) = parsed else {
            return;
        };

        let performance = web_sys::window().unwrap().performance().unwrap();
        let start_time = performance.now();

        database.with_value(|db| {
            let data = db.as_ref().unwrap();
            let results = match mode {
                SearchMode::Sell => {
                    execute_sell_mode(data, &ap_sel, &custom_ac, &user_settings, &gm, &parsed)
                }
                SearchMode::AnyDestination | SearchMode::SpecificDestination => {
                    match execute_scheduled_mode(
                        data,
                        &ap_sel,
                        &ap_dest,
                        mode,
                        &custom_ac,
                        &user_settings,
                        &gm,
                        &parsed,
                    ) {
                        Some(v) => v,
                        None => {
                            set_routes.set(vec![]);
                            set_stats.set(RouteStats::default());
                            return;
                        }
                    }
                }
            };

            set_stats.set(RouteStats {
                count: results.len(),
                time_ms: performance.now() - start_time,
            });
            set_routes.set(results);
        });
    });

    view! {
        <div class="route-options">
            <Show when=move || show_destination_input.get()>
                <APSearch selected=ap_destination active=ap_destination_active />
            </Show>

            <Show when=move || show_distance_filters.get()>
                <FilterInput
                    label="Distance (km)"
                    min_val=dist_min
                    max_val=dist_max
                    error=dist_error
                />
                <FilterInput label="Time (hr)" min_val=ft_min max_val=ft_max error=ft_error />
            </Show>

            <Show when=move || show_num_ac.get()>
                <StrategyInput
                    label="AC per route"
                    value=num_ac_input
                    is_active=num_ac_max
                    active_label="Maximise"
                    error=num_ac_error
                />
            </Show>

            <Show when=move || show_tpd.get()>
                <StrategyInput
                    label="Trips per day per AC"
                    value=tpd_input
                    is_active=tpd_max
                    active_label="Maximise"
                    error=tpd_error
                />
            </Show>

            <Show when=move || show_ci.get()>
                <StrategyInput
                    label="Cost Index"
                    value=ci_input
                    is_active=ci_auto
                    active_label="Align Max Time"
                    error=ci_error
                    disable_button=Signal::derive(move || {
                        show_destination_input.get() || ft_max.get().trim().is_empty()
                    })
                />
                <ConfigAlgoInput algo=config_algo ac_type=ac_type />
            </Show>

            <Show when=move || search_mode.get() == SearchMode::AnyDestination>
                <label class:disabled=move || {
                    dist_max.get().trim().is_empty()
                }>
                    "Inflate distance"
                    <input
                        type="checkbox"
                        prop:checked=stopover_mode
                        prop:disabled=move || dist_max.get().trim().is_empty()
                        on:change=move |ev| stopover_mode.set(event_target_checked(&ev))
                    />
                </label>
                <div class="toggle-group">
                    <span>"Sort"</span>
                    <div class="toggle-options">
                        <button
                            class:active=move || sort_by.get() == SortBy::ProfitPerAcPerDay
                            on:click=move |_| sort_by.set(SortBy::ProfitPerAcPerDay)
                        >
                            "$/d/ac"
                        </button>
                        <button
                            class:active=move || sort_by.get() == SortBy::ProfitPerTrip
                            on:click=move |_| sort_by.set(SortBy::ProfitPerTrip)
                        >
                            "$/t"
                        </button>
                    </div>
                </div>
            </Show>
        </div>
    }
}

#[component]
pub fn RouteList(
    #[prop(into)] routes: ReadSignal<Vec<WebRoute>>,
    #[prop(into)] stats: ReadSignal<RouteStats>,
    #[prop(into)] show_origin: Signal<bool>,
    #[prop(into)] search_mode: Signal<SearchMode>,
) -> impl IntoView {
    let settings = expect_context::<RwSignal<Settings>>();
    let page = RwSignal::new(0usize);
    let page_size = 10usize;

    Effect::new(move |_| {
        routes.track();
        page.set(0);
    });

    let paged_results = Memo::new(move |_| {
        let p = page.get();
        routes.with(|all| {
            let start = p * page_size;
            if start >= all.len() {
                return vec![];
            }
            let end = (start + page_size).min(all.len());
            all[start..end].to_vec()
        })
    });

    let total_pages = Memo::new(move |_| {
        let s = stats.get();
        if s.count == 0 {
            1
        } else {
            s.count.div_ceil(page_size)
        }
    });

    view! {
        <Show when=move || {
            let s = stats.get();
            s.count > 0
        }>
            <div class="results-meta">
                <span>
                    {move || stats.get().count}
                    {move || {
                        if search_mode.get() == SearchMode::Sell {
                            " ferry routes found in "
                        } else {
                            " routes found in "
                        }
                    }} {move || format!("~{:.1}", stats.get().time_ms)} "ms"
                </span>
                <div class="pagination">
                    <button
                        class="download"
                        title="Download CSV"
                        on:click=move |_| {
                            download_csv(routes.get(), settings.get().csv_time_format);
                        }
                    >
                        <DownloadIcon size=16 />
                        "CSV"
                    </button>
                    <button
                        disabled=move || page.get() == 0
                        on:click=move |_| page.update(|p| *p -= 1)
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="16"
                            height="16"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            stroke-width="2"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                        >
                            <path d="m15 18-6-6 6-6" />
                        </svg>
                    </button>
                    <span>{move || page.get() + 1} " / " {total_pages}</span>
                    <button
                        disabled=move || { page.get() + 1 == total_pages.get() }
                        on:click=move |_| page.update(|p| *p += 1)
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="16"
                            height="16"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            stroke-width="2"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                        >
                            <path d="m9 18 6-6-6-6" />
                        </svg>
                    </button>
                </div>
            </div>
        </Show>

        <div class="results-list">
            <For
                each=move || paged_results.get()
                key=|_| uuid::Uuid::new_v4()
                children=move |route| {
                    match route {
                        WebRoute::Scheduled(r) => {
                            view! { <RouteCard route=r show_origin=show_origin /> }.into_any()
                        }
                        WebRoute::Ferry(r) => {
                            view! { <FerryRouteCard route=r show_origin=show_origin /> }.into_any()
                        }
                    }
                }
            />
        </div>
    }
}

#[derive(Clone)]
struct StatItem {
    label: &'static str,
    val: String,
    class: &'static str,
}

fn airport_code(settings: RwSignal<Settings>, airport: &Airport) -> Signal<String> {
    let iata = airport.iata.to_string();
    let icao = airport.icao.to_string();
    Signal::derive(move || {
        settings.with(|s| match s.airport_code_pref {
            AirportCodePref::Iata => iata.clone(),
            AirportCodePref::Icao => icao.clone(),
        })
    })
}

#[component]
pub fn RouteCard(route: WebScheduledRoute, show_origin: Signal<bool>) -> impl IntoView {
    let settings = expect_context::<RwSignal<Settings>>();
    let fmt_money = |v: f32| format!("$ {}", format_thousands(v));
    let fmt_dist = |d: f32| format!("{} km", format_thousands(d));

    let is_cargo = matches!(route.config, ConfigVariant::Cargo(_));
    let ft_str = format_flight_time_hms(route.flight_time);

    let dist_diff_pct = (route.total_distance.get() - route.direct_distance.get())
        / route.direct_distance.get()
        * 100.0;
    let dist_pct_str = if dist_diff_pct > 0.1 {
        format!(", +{:.1}%", dist_diff_pct)
    } else {
        String::new()
    };

    let origin_code = airport_code(settings, &route.origin);
    let dest_code = airport_code(settings, &route.destination);

    let render_row = |label: &'static str, items: Vec<StatItem>| {
        view! {
            <span class="label">{label}</span>
            <div class="stat-group">
                {items
                    .into_iter()
                    .map(|item| {
                        view! {
                            <div class="stat-pair">
                                <span class=format!("letter {}", item.class)>{item.label}</span>
                                <span class="val">{item.val}</span>
                            </div>
                        }
                    })
                    .collect::<Vec<_>>()}
            </div>
        }
    };

    let demand_stats = if is_cargo {
        let d: CargoDemand = (&route.demand).into();
        vec![
            StatItem {
                label: "L",
                val: format_thousands(d.l),
                class: "l",
            },
            StatItem {
                label: "H",
                val: format_thousands(d.h),
                class: "h",
            },
        ]
    } else {
        vec![
            StatItem {
                label: "Y",
                val: format_thousands(route.demand.y),
                class: "y",
            },
            StatItem {
                label: "J",
                val: format_thousands(route.demand.j),
                class: "j",
            },
            StatItem {
                label: "F",
                val: format_thousands(route.demand.f),
                class: "f",
            },
        ]
    };

    let contribution_str = format!("C${:.0}/f", route.contribution);

    let config_stats = match route.config {
        ConfigVariant::Cargo(c) => vec![
            StatItem {
                label: "L",
                val: format!("{}%", c.l),
                class: "l",
            },
            StatItem {
                label: "H",
                val: format!("{}%", c.h),
                class: "h",
            },
        ],
        ConfigVariant::Pax(c) => vec![
            StatItem {
                label: "Y",
                val: c.y.to_string(),
                class: "y",
            },
            StatItem {
                label: "J",
                val: c.j.to_string(),
                class: "j",
            },
            StatItem {
                label: "F",
                val: c.f.to_string(),
                class: "f",
            },
        ],
    };

    let ticket_stats = match route.ticket {
        Ticket::Cargo(t) => vec![
            StatItem {
                label: "L",
                val: format!("${:.2}", t.l),
                class: "l",
            },
            StatItem {
                label: "H",
                val: format!("${:.2}", t.h),
                class: "h",
            },
        ],
        Ticket::Pax(t) | Ticket::VIP(t) => vec![
            StatItem {
                label: "Y",
                val: format_thousands(t.y),
                class: "y",
            },
            StatItem {
                label: "J",
                val: format_thousands(t.j),
                class: "j",
            },
            StatItem {
                label: "F",
                val: format_thousands(t.f),
                class: "f",
            },
        ],
    };

    let max_tpd = (24.0 / route.flight_time.get()).floor() as u8;
    let tpd_warning = route.trips_per_day > max_tpd;

    let revenue_str = fmt_money(route.revenue);
    let fuel_price = settings.get().fuel_price.get();
    let co2_price = settings.get().co2_price.get();
    let fuel_cost = route.fuel * fuel_price / 1000.0;
    let co2_cost = route.co2 * co2_price / 1000.0;
    let fuel_cost_str = fmt_money(fuel_cost);
    let co2_cost_str = fmt_money(co2_cost);
    let fuel_amount_str = format_thousands(route.fuel);
    let co2_amount_str = format_thousands(route.co2);
    let acheck_cost_str = fmt_money(route.acheck_cost);
    let repair_cost_str = fmt_money(route.repair_cost);

    view! {
        <div class="route-card">
            <div class="header">
                <Show when=move || show_origin.get()>
                    <div class="dest-block">
                        <span class="label">"from"</span>
                        <div class="codes">
                            <span class="main">{origin_code}</span>
                        </div>
                        <div class="name">
                            {route.origin.name.to_string()} ", " {route.origin.country.to_string()}
                        </div>
                    </div>
                </Show>
                {route
                    .stopover
                    .map(|stop| {
                        let stop_code = airport_code(settings, &stop);
                        view! {
                            <div class="dest-block">
                                <span class="label">"via"</span>
                                <div class="codes">
                                    <span class="main">{stop_code}</span>
                                </div>
                                <div class="name">
                                    {stop.name.to_string()} ", " {stop.country.to_string()}
                                </div>
                            </div>
                        }
                    })}
                <div class="dest-block">
                    <span class="label">"to"</span>
                    <div class="codes">
                        <span class="main">{dest_code}</span>
                    </div>
                    <div class="name">
                        {route.destination.name.to_string()} ", "
                        {route.destination.country.to_string()}
                    </div>
                </div>
            </div>

            <div class=format!(
                "grid-info {}",
                if is_cargo { "cargo" } else { "" },
            )>
                {render_row("Demand", demand_stats)} {render_row("Config", config_stats)}
                {render_row("Tickets", ticket_stats)}
            </div>

            <div class="details-text">
                {fmt_dist(route.total_distance.get())} " (" {ft_str} {dist_pct_str} ")" " ⋅ CI="
                {route.ci.get()} " ⋅ " {contribution_str.clone()} <br /> {route.trips_per_day}
                " t/d/ac" " × " {route.num_aircraft} " ac" " ⋅ " {fmt_money(route.profit)} "/t"
                " ⋅ " {fmt_money(route.profit * route.trips_per_day as f32)} "/d/ac"
            </div>
            <Show when=move || tpd_warning>
                <div class="warning-text">
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                    >
                        <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"></path>
                        <path d="M12 9v4"></path>
                        <path d="M12 17h.01"></path>
                    </svg>
                    <span>{format!("Exceeds 24hrs (max: {} t/d)", max_tpd)}</span>
                </div>
            </Show>

            <details class="cost-breakdown">
                <summary>
                    <svg
                        class="expand-arrow"
                        xmlns="http://www.w3.org/2000/svg"
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                    >
                        <path d="m6 9 6 6 6-6" />
                    </svg>
                </summary>
                <div class="cost-grid">
                    <span class="cost-label">"Revenue"</span>
                    <span class="cost-value revenue">{revenue_str}</span>
                    <span class="cost-label">"Fuel (" {fuel_amount_str.clone()} " lbs)"</span>
                    <span class="cost-value expense">{fuel_cost_str}</span>
                    <span class="cost-label">"CO₂ (" {co2_amount_str.clone()} " quota)"</span>
                    <span class="cost-value expense">{co2_cost_str}</span>
                    <span class="cost-label">"A-check"</span>
                    <span class="cost-value expense">{acheck_cost_str}</span>
                    <span class="cost-label">"Repair"</span>
                    <span class="cost-value expense">{repair_cost_str}</span>
                </div>
            </details>
        </div>
    }
}

#[component]
pub fn FerryRouteCard(route: WebFerryRoute, show_origin: Signal<bool>) -> impl IntoView {
    let settings = expect_context::<RwSignal<Settings>>();
    let fmt_money = |v: f32| format!("$ {}", format_thousands(v));
    let fmt_dist = |d: f32| format!("{} km", format_thousands(d));

    let ft_str = format_flight_time_hms(route.flight_time);

    let origin_code = airport_code(settings, &route.origin);
    let dest_code = airport_code(settings, &route.destination);

    let fuel_price = settings.get().fuel_price.get();
    let fuel_cost = route.fuel * fuel_price / 1000.0;

    view! {
        <div class="route-card">
            <div class="header">
                <Show when=move || show_origin.get()>
                    <div class="dest-block">
                        <span class="label">"from"</span>
                        <div class="codes">
                            <span class="main">{origin_code}</span>
                        </div>
                        <div class="name">
                            {route.origin.name.to_string()} ", " {route.origin.country.to_string()}
                        </div>
                    </div>
                </Show>
                <div class="dest-block">
                    <span class="label">"sell at"</span>
                    <div class="codes">
                        <span class="main">{dest_code}</span>
                    </div>
                    <div class="name">
                        {route.destination.name.to_string()} ", "
                        {route.destination.country.to_string()}
                    </div>
                </div>
            </div>

            <div class="details-text">
                {fmt_dist(route.direct_distance.get())} " (" {ft_str} ")" " ⋅ Market: "
                {route.destination.market} "%" <br /> "Fuel: " {fmt_money(fuel_cost)} " ("
                {format_thousands(route.fuel)} " lbs)" " ⋅ Profit: " {fmt_money(route.profit)}
            </div>
        </div>
    }
}
