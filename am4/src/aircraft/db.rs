/*!
Implements an in-memory, indexed aircraft database.

An aircraft has unique identifiers (such as [Id], [ShortName], [Name] etc.)
and we wish to query it in O(1) given the primary key.

## Database building
Given &[[Aircraft]], build hashmaps that point [SearchKey]s to the array index.

## Searching
1. Given user input (&[str]), e.g. `shortname:b744[sfc]`, `B74[1]`
2. Parse into a [QueryCtx]

    a. The [QueryKey]: e.g. ShortName("b744"), All("B74")

    b. The [Modification]: e.g. speed+fuel+co2, engine variant 1
3. Convert it into the primary key.
    - in the case of [QueryKey::All] (general search), attempt all [SearchKey]s.
4. Dereference the associated array index to get the aircraft.
5. If engine is specified, choose the correct [AircraftVariants].

Apply the modifiers and return a [CustomAircraft].
*/
use crate::aircraft::custom::{CustomAircraft, Modification};
use crate::aircraft::{Aircraft, AircraftError, EnginePriority, Id, Name, ShortName};
use crate::utils::{queue_suggestions, Suggestion, MAX_SUGGESTIONS};
use jaro_winkler::jaro_winkler;
use std::collections::{BinaryHeap, HashMap};
use std::str::FromStr;
use thiserror::Error;

#[cfg(feature = "rkyv")]
use crate::utils::ParseError;
#[cfg(feature = "rkyv")]
use rkyv::{self, Deserialize};

pub static LENGTH_MAX: f32 = 77.0;
pub static LENGTH_MEAN: f32 = 34.278454;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SearchKey {
    Id(Id),
    ShortName(ShortName),
    Name(Name),
}

impl From<Id> for SearchKey {
    fn from(id: Id) -> Self {
        SearchKey::Id(id)
    }
}

impl From<ShortName> for SearchKey {
    fn from(sn: ShortName) -> Self {
        SearchKey::ShortName(ShortName(sn.0.to_uppercase()))
    }
}

impl From<Name> for SearchKey {
    fn from(name: Name) -> Self {
        SearchKey::Name(Name(name.0.to_uppercase()))
    }
}

#[derive(Debug, Clone)]
pub enum QueryKey {
    All(String),
    Id(Id),
    ShortName(ShortName),
    Name(Name),
}

#[derive(Debug, Clone)]
pub struct QueryCtx {
    pub key: QueryKey,
    pub modifiers: Modification,
}

impl FromStr for QueryCtx {
    type Err = AircraftSearchError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (s, mods) = match s.trim().split_once('[') {
            None => (s, Modification::default()),
            Some((s, mods_str)) => {
                if let Some(end_idx) = mods_str.find(']') {
                    (s, Modification::from_str(&mods_str[..end_idx])?)
                } else {
                    return Err(AircraftSearchError::MissingClosingBracket);
                }
            }
        };

        if s.is_empty() {
            return Err(AircraftSearchError::EmptyQuery);
        }

        let key = match s.to_uppercase().split_once(':') {
            None => QueryKey::All(s.to_string()),
            Some((k, v)) => match k {
                "SHORTNAME" => QueryKey::ShortName(ShortName(v.to_string())),
                "NAME" => QueryKey::Name(Name(v.to_string())),
                "ID" => QueryKey::Id(Id::from_str(v)?),
                v => return Err(AircraftSearchError::InvalidColumnSpecifier(v.to_string())),
            },
        };

        Ok(QueryCtx {
            key,
            modifiers: mods,
        })
    }
}

impl From<&QueryCtx> for Result<SearchKey, AircraftSearchError> {
    fn from(ctx: &QueryCtx) -> Self {
        match &ctx.key {
            QueryKey::All(s) => {
                if let Ok(v) = Id::from_str(s) {
                    Ok(SearchKey::from(v))
                } else if let Ok(v) = ShortName::from_str(s) {
                    Ok(SearchKey::from(v))
                } else if let Ok(v) = Name::from_str(s) {
                    Ok(SearchKey::from(v))
                } else {
                    // all parsers failed, we can be sure it does not exist in the database
                    Err(AircraftSearchError::AircraftNotFound(ctx.clone()))
                }
            }
            QueryKey::Id(id) => Ok(SearchKey::from(*id)),
            QueryKey::ShortName(sn) => Ok(SearchKey::from(sn.clone())),
            QueryKey::Name(name) => Ok(SearchKey::from(name.clone())),
        }
    }
}

pub type AircraftVariants = HashMap<EnginePriority, usize>;

/// An immutable collection of aircrafts, stored entirely in-memory.
///
/// This must be created from a rkyv archive of a [`Vec<Aircraft>`] via [Self::from_bytes].
#[derive(Debug)]
pub struct Aircrafts {
    data: Vec<Aircraft>,
    index: HashMap<SearchKey, AircraftVariants>,
}

impl Aircrafts {
    #[cfg(feature = "rkyv")]
    pub fn from_bytes(buffer: &[u8]) -> Result<Self, ParseError> {
        let archived = rkyv::check_archived_root::<Vec<Aircraft>>(buffer)
            .map_err(|e| ParseError::ArchiveError(e.to_string()))?;

        let data: Vec<Aircraft> = archived
            .deserialize(&mut rkyv::Infallible)
            .map_err(|e| ParseError::DeserialiseError(e.to_string()))?;

        let mut index = HashMap::<SearchKey, AircraftVariants>::new();

        for (i, ac) in data.iter().enumerate() {
            index
                .entry(SearchKey::from(ac.id))
                .or_default()
                .insert(ac.priority, i);

            index
                .entry(SearchKey::from(ac.shortname.clone()))
                .or_default()
                .insert(ac.priority, i);

            index
                .entry(SearchKey::from(ac.name.clone()))
                .or_default()
                .insert(ac.priority, i);
        }

        Ok(Self { data, index })
    }

    /// Search for an aircraft, defaulting to the engine that gives the fastest speed, (priority == 0, fastest)
    pub fn search(&self, s: &str) -> Result<CustomAircraft, AircraftSearchError> {
        let ctx = QueryCtx::from_str(s)?;
        let engines = self.search_by_ctx(&ctx)?;

        if let Some(i) = engines.get(&ctx.modifiers.engine) {
            Ok(CustomAircraft::from_aircraft_and_modifiers(
                self.data[*i].clone(),
                ctx.modifiers,
            ))
        } else {
            // we got the suggestions for free, so we might as well return in the error
            Err(AircraftSearchError::EnginePriorityNotFound {
                src: ctx.modifiers.engine,
                suggestions: engines.keys().cloned().collect(),
            })
        }
    }

    /// Search all engine variants for a given aircraft
    pub fn search_engines(&self, s: &str) -> Result<&AircraftVariants, AircraftSearchError> {
        let ctx = QueryCtx::from_str(s)?;
        self.search_by_ctx(&ctx)
    }

    pub fn suggest(&self, s: &str) -> Result<Vec<Suggestion<&Aircraft>>, AircraftSearchError> {
        let ctx = QueryCtx::from_str(s)?;
        self.suggest_by_ctx(&ctx)
    }

    fn search_by_ctx(&self, ctx: &QueryCtx) -> Result<&AircraftVariants, AircraftSearchError> {
        let key = Result::<SearchKey, AircraftSearchError>::from(ctx)?;

        self.index
            .get(&key)
            .ok_or(AircraftSearchError::AircraftNotFound(ctx.clone()))
    }

    pub fn suggest_by_ctx(
        &self,
        ctx: &QueryCtx,
    ) -> Result<Vec<Suggestion<&Aircraft>>, AircraftSearchError> {
        // TODO: this is a hack to get the uppercase version of the parsed query
        let key = Result::<SearchKey, AircraftSearchError>::from(ctx)?;
        let su = match key {
            SearchKey::ShortName(v) => v.0,
            SearchKey::Name(v) => v.0,
            SearchKey::Id(v) => v.0.to_string(),
        };

        let mut heap = BinaryHeap::with_capacity(MAX_SUGGESTIONS);

        for (key, variants) in &self.index {
            // only search first engine variant
            // TODO: restrict to only search by shortname if ctx.key is shortname
            if let Some(i) = variants.values().next() {
                let s = match key {
                    SearchKey::ShortName(v) => &v.0,
                    SearchKey::Name(v) => &v.0,
                    _ => continue, // ignore searching by id
                };
                let similarity = jaro_winkler(s, &su);
                queue_suggestions(&mut heap, &self.data[*i], similarity);
            }
        }

        Ok(heap.into_sorted_vec())
    }

    pub fn data(&self) -> &Vec<Aircraft> {
        &self.data
    }

    pub fn index(&self) -> &HashMap<SearchKey, AircraftVariants> {
        &self.index
    }
}

#[derive(Debug, Error)]
pub enum AircraftSearchError {
    #[error("Empty query")] // TODO: "[sfc]" will match cause this.
    EmptyQuery,
    #[error("Invalid column specifier: `{0}`. Did you mean: `shortname`, `name` or `id`?")]
    InvalidColumnSpecifier(String),
    #[error("Aircraft `{0:?}` not found")]
    AircraftNotFound(QueryCtx),
    #[error("Engine with priority `{src}` does not exist for this aircraft. Did you mean: {}?", .suggestions.iter().map(|p| format!("`{p}`")).collect::<Vec<String>>().join(", "))]
    EnginePriorityNotFound {
        src: EnginePriority,
        suggestions: Vec<EnginePriority>,
    },
    #[error("Found opening `[` but no closing `]`.")]
    MissingClosingBracket,
    #[error(transparent)]
    AircraftField(#[from] AircraftError),
}
