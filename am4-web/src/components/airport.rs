use crate::components::format_thousands;
use crate::components::search::MultiSelect;
use crate::db::Data;
use am4::airport::Airport;
use am4::user::{AirportCodePref, Settings};
use leptos::prelude::*;

#[allow(non_snake_case)]
#[component]
pub fn APSearch(
    #[prop(into)] selected: RwSignal<Vec<Airport>>,
    #[prop(into)] active: RwSignal<Option<Airport>>,
    #[prop(optional)] label: Option<&'static str>,
) -> impl IntoView {
    let database = expect_context::<StoredValue<Option<Data>>>();

    let search = Callback::new(move |q: String| {
        database.with_value(|db| {
            db.as_ref()
                .unwrap()
                .airports
                .suggest(&q)
                .unwrap_or_default()
                .into_iter()
                .map(|s| s.item.clone())
                .collect()
        })
    });

    let render_option =
        move |ap: Airport,
              _idx: usize,
              _suggestions: Memo<Vec<Airport>>,
              _highlight_idx: ReadSignal<usize>,
              _select: Callback<Airport>,
              _update: Callback<Airport>,
              _update_all: Callback<Box<dyn Fn(Airport) -> Airport + Send + Sync>>| {
            view! {
                <div class="ap-option">
                    <div class="details">
                        <div class="row main">
                            <span class="codes">
                                {ap.iata.to_string()}" / "{ap.icao.to_string()}
                            </span>
                            <span class="sep">" / "</span>
                            <span class="location">
                                {ap.country.to_string()}", "{ap.name.to_string()}
                            </span>
                        </div>
                        <div class="row info">
                            <span class="stat-val rwy">{format_thousands(ap.rwy)}" ft"</span>
                            <span class="cdot">" ⋅ "</span>

                            <span class="stat-val market">{ap.market}"%"</span>
                            <span class="cdot">" ⋅ "</span>

                            <span class="stat-val hub-cost">
                                "$ "{format_thousands(ap.hub_cost)}
                            </span>
                        </div>
                    </div>
                </div>
            }
            .into_any()
        };

    let render_pill = move |ap: Airport, remove: Callback<()>| {
        let settings = expect_context::<RwSignal<Settings>>();
        let iata = ap.iata.to_string();
        let icao = ap.icao.to_string();
        view! {
            <div class="ap-pill">
                <span class="code">
                    {move || {
                        if settings.with(|s| s.airport_code_pref == AirportCodePref::Icao) {
                            icao.clone()
                        } else {
                            iata.clone()
                        }
                    }}
                </span>
                <span
                    class="remove"
                    on:click=move |ev| {
                        ev.stop_propagation();
                        remove.run(());
                    }
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                    >
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </span>
            </div>
        }
        .into_any()
    };

    let parse_token = Callback::new(move |token: String| {
        database.with_value(|db| {
            db.as_ref()
                .and_then(|db| db.airports.search(&token).ok())
                .cloned()
        })
    });

    let serialize = Callback::new(|ap: Airport| ap.icao.to_string());
    let key = Callback::new(|ap: Airport| ap.idx);

    view! {
        <div class="ap-search">
            <Show when=move || label.is_some_and(|s| !s.is_empty())>
                <label>{label.unwrap_or_default()}</label>
            </Show>
            <MultiSelect
                selected=selected
                active=active
                search=search
                placeholder="Search airports..."
                render_option=render_option
                render_pill=render_pill
                serialize=serialize
                key=key
                parse_token=parse_token
            />
        </div>
    }
}

/// Component that renders the airport details card
/// Must be rendered inside a component that has APSearch as an ancestor
#[allow(non_snake_case)]
#[component]
pub fn APDetails() -> impl IntoView {
    let active = expect_context::<RwSignal<Option<Airport>>>();

    view! { {move || active.get().map(|ap| view! { <Ap airport=ap /> })} }
}

#[allow(dead_code)]
#[allow(non_snake_case)]
#[component]
fn Ap(airport: Airport) -> impl IntoView {
    view! {
        <div class="ap-card">
            <h3>
                {airport.name.to_string()} ", " {airport.country.to_string()} " ("
                {airport.iata.to_string()} " / " {airport.icao.to_string()} ")"
            </h3>
            <table>
                <tr>
                    <th>"Full Name"</th>
                    <td>{airport.fullname.to_string()}</td>
                </tr>
                <tr>
                    <th>"Continent"</th>
                    <td>{airport.continent.to_string()}</td>
                </tr>
                <tr>
                    <th>"Location"</th>
                    <td>{format!("{}, {}", airport.location.lat, airport.location.lng)}</td>
                </tr>
                <tr>
                    <th>"Runway"</th>
                    <td>{format_thousands(airport.rwy)}" ft"</td>
                </tr>
                <tr>
                    <th>"Market"</th>
                    <td>{airport.market}"%"</td>
                </tr>
                <tr>
                    <th>"Hub Cost"</th>
                    <td>"$ "{format_thousands(airport.hub_cost)}</td>
                </tr>
                <tr>
                    <th>"Runway Codes"</th>
                    <td>{airport.rwy_codes.join(",")}</td>
                </tr>
            </table>
        </div>
    }
}
