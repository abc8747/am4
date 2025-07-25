use am4::aircraft::db::Aircrafts;
use am4::airport::{db::Airports, Airport};
use am4::route::db::DistanceMatrix;
use am4::{AC_FILENAME, AP_FILENAME, DIST_FILENAME};
use indexed_db_futures::database::Database;
use indexed_db_futures::error::OpenDbError;
use indexed_db_futures::prelude::*;
use indexed_db_futures::transaction::TransactionMode;
use leptos::{
    logging::log,
    wasm_bindgen::{prelude::*, JsValue},
    web_sys,
};
use thiserror::Error;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    js_sys::{Array, Uint8Array},
    window, Blob, BlobPropertyBag, Response,
};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Idb {
    database: Database,
}

// TODO: replace unwrap with proper handling with rkvy parse errors.
#[allow(dead_code)]
impl Idb {
    /// Main database name
    const NAME_DB: &str = "am4help";
    /// Object store name for all data
    const NAME_STORE: &str = "data";

    /// connect to the database and ensure that `am4help/data` object store exists
    pub async fn connect() -> Result<Self, OpenDbError> {
        let database = Database::open(Self::NAME_DB)
            .with_on_upgrade_needed(|_event, db| {
                if !db.object_store_names().any(|n| n == Self::NAME_STORE) {
                    let _ = db.create_object_store(Self::NAME_STORE);
                    todo!()
                }
                Ok(())
            })
            .await?;
        Ok(Self { database })
    }

    pub async fn clear(&self) -> Result<(), indexed_db_futures::error::Error> {
        let tx = self
            .database
            .transaction(Self::NAME_STORE)
            .with_mode(TransactionMode::Readwrite)
            .build()?;
        tx.object_store(Self::NAME_STORE)?.clear()?;
        Ok(())
    }

    /// Load a binary file from the IndexedDb. If the blob is not found*,
    /// fetch it from the server and cache it in the IndexedDb.
    /// not found: `Response`* -> `Blob`* -> IndexedDb -> `Blob` -> `ArrayBuffer` -> `Vec<u8>`
    pub async fn get_blob(&self, k: &str, url: &str) -> Result<Vec<u8>, DatabaseError> {
        let tx = self
            .database
            .transaction(Self::NAME_STORE)
            .with_mode(TransactionMode::Readwrite)
            .build()?;
        let store = tx.object_store(Self::NAME_STORE)?;
        let jsb = match store.get(k).primitive()?.await? {
            Some(b) => b,
            None => {
                let b = fetch_bytes(url).await?;
                store.put(&b).with_key(k).await?;
                tx.commit().await?;
                b
            }
        };
        let ab = JsFuture::from(jsb.dyn_into::<Blob>()?.array_buffer()).await?;
        Ok(Uint8Array::new(&ab).to_vec())
    }

    /// Load the flat distances from the indexeddb. If the blob is not found*,
    /// generate it from the slice of airports and cache it in the indexeddb.
    /// not found: `&[Airport]`* -> `Distances`* (return this) -> `Blob`* -> IndexedDb
    /// found: IndexedDb -> `Blob` -> `ArrayBuffer` -> `Distances`
    async fn get_distances(&self, aps: &[Airport]) -> Result<DistanceMatrix, DatabaseError> {
        let tx = self
            .database
            .transaction(Self::NAME_STORE)
            .with_mode(TransactionMode::Readwrite)
            .build()?;
        let store = tx.object_store(Self::NAME_STORE)?;
        match store
            .get::<JsValue, _, _>(DIST_FILENAME)
            .primitive()?
            .await?
        {
            Some(b) => {
                let ab = JsFuture::from(b.dyn_into::<Blob>()?.array_buffer()).await?;
                let bytes = Uint8Array::new(&ab).to_vec();
                Ok(DistanceMatrix::from_bytes(&bytes).unwrap())
            }
            None => {
                let distances = DistanceMatrix::from_airports(aps);
                let b = distances.to_bytes().unwrap();

                // https://github.com/rustwasm/wasm-bindgen/issues/1693
                // effectively, this is `new Blob([new Uint8Array(b)], {type: 'application/octet-stream'})`
                let ja = Array::new();
                ja.push(&Uint8Array::from(b.as_slice()).buffer());
                let opts = BlobPropertyBag::new();
                opts.set_type("application/octet-stream");
                let blob = Blob::new_with_u8_array_sequence_and_options(&ja, &opts)?;
                let _ = store.put(&blob).with_key(DIST_FILENAME);
                Ok(distances)
            }
        }
    }

    pub async fn _init_db(&self) -> Result<Data, DatabaseError> {
        let bytes = self
            .get_blob(AP_FILENAME, format!("assets/{AP_FILENAME}").as_str())
            .await?;
        let airports = Airports::from_bytes(&bytes).unwrap();
        log!("airports: {}", airports.data().len());

        let bytes = self
            .get_blob(AC_FILENAME, format!("assets/{AC_FILENAME}").as_str())
            .await?;
        let aircrafts = Aircrafts::from_bytes(&bytes).unwrap();
        log!("aircrafts: {}", aircrafts.data().len());

        let distances = self.get_distances(airports.data()).await?;
        log!("distances: {}", distances.data().len());
        // let distances = Distances::from_airports(&(airports.data()));
        // let bytes = self.fetch("routes", "assets/routes.bin", set_progress).await?;
        // set_progress(LoadDbProgress::Parsing("routes".to_string()));
        // let routes = Routes::from_bytes(&bytes).unwrap();
        // log!("routes: {}", routes.data().len());

        // set_progress(LoadDbProgress::Loaded);
        Ok(Data {
            aircrafts,
            airports,
        })
    }
}

async fn fetch_bytes(path: &str) -> Result<JsValue, DatabaseError> {
    let window = window().unwrap();
    let resp_value = JsFuture::from(window.fetch_with_str(path)).await?;
    let resp = resp_value.dyn_into::<Response>()?;
    let jsb = JsFuture::from(resp.blob()?).await?;
    Ok(jsb)
}

#[allow(dead_code)]
pub struct Data {
    pub aircrafts: Aircrafts,
    pub airports: Airports,
}

#[allow(dead_code)]
#[derive(Clone, PartialEq)]
pub enum LoadDbProgress {
    Starting,
    IDBConnect,
    IDBRead(String),
    IDBWrite(String),
    Fetching(String),
    Parsing(String),
    Loaded,
    Err,
}

#[derive(Error, Debug, PartialEq)]
pub enum DatabaseError {
    #[error("IDB error: {:?}", self)]
    Idb(#[from] indexed_db_futures::error::Error),
    #[error("DOM exception: {:?}", self)]
    Dom(web_sys::DomException),
    #[error("JavaScript exception: {:?}", self)]
    Js(JsValue),
}

impl From<web_sys::DomException> for DatabaseError {
    fn from(dom_exception: web_sys::DomException) -> Self {
        Self::Dom(dom_exception)
    }
}

impl From<JsValue> for DatabaseError {
    fn from(js_value: JsValue) -> Self {
        Self::Js(js_value)
    }
}
