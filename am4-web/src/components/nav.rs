use crate::db::LoadDbProgress;
use leptos::prelude::*;

static VERSION: &str = env!("CARGO_PKG_VERSION");

impl std::fmt::Display for LoadDbProgress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadDbProgress::Starting => write!(f, "starting"),
            LoadDbProgress::IDBConnect => write!(f, "idb(connect)"),
            LoadDbProgress::IDBRead(key) => write!(f, "idb(read): {}", key),
            LoadDbProgress::IDBWrite(key) => write!(f, "idb(write): {}", key),
            LoadDbProgress::Fetching(key) => write!(f, "fetch: {}", key),
            LoadDbProgress::Parsing(key) => write!(f, "parse: {}", key),
            LoadDbProgress::Loaded => write!(f, ""),
            LoadDbProgress::Err => write!(f, "error"),
        }
    }
}

#[component]
#[allow(non_snake_case)]
pub fn Header() -> impl IntoView {
    // let prog_str = move || progress.get().to_string();

    let clear_db = Action::new(|_| async {
        todo!();
        // Idb::connect().await.unwrap().clear().await.unwrap();
    });

    view! {
        <header role="banner">
            <div id="global-nav">
                <a href="https://abc8747.github.io/am4/" target="_blank">
                    <img src="/assets/img/logo-64.webp" alt="logo" height="32" width="32" />
                </a>
                <div>
                    <span id="name">AM4Help</span>
                    <span id="version">" v" {VERSION}</span>
                </div>
            </div>
            <div id="local-bar">
                <nav>
                    <ul>
                        <li>
                            <b>
                                <a href="/">"Home"</a>
                            </b>
                        </li>
                        <li>"Console"</li>
                        <li>
                            <a href="https://abc8747.github.io/am4/formulae/">"Formulae"</a>
                        </li>
                        <li
                            id="clear-database"
                            on:click=move |_| {
                                clear_db.dispatch(&());
                            }
                        >

                            "Clear Database"
                        </li>
                    </ul>
                </nav>
            </div>
        </header>
    }
}
