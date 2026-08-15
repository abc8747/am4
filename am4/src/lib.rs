/*!
 * Core tools and utilities for the game Airline Manager 4.
 *
 * We do not use external databases, but instead create immutable, in-memory
 * stores of all aircrafts, airports and route demands for maximum performance.
 *
 * It is intended for use as a core library in many targets, including WASM, CLI,
 * Web API and a discord bot.
 */

#![warn(missing_debug_implementations)]

pub mod aircraft;
pub mod airport;
pub mod utils;

pub mod campaign;
pub mod user;

pub mod route;

// to keep track of changes for data files in `../data`
#[macro_export]
macro_rules! ac_version {
    () => {
        "9"
    };
}
pub const AC_FILENAME: &str = concat!("aircrafts-v", ac_version!(), ".bin");

#[macro_export]
macro_rules! ap_version {
    () => {
        "2"
    };
}
pub const AP_FILENAME: &str = concat!("airports-v", ap_version!(), ".bin");
// distances are generated from the airports
pub const DIST_FILENAME: &str = concat!("distances-v", ap_version!(), ".bin");

#[macro_export]
macro_rules! demand_version {
    () => {
        "1"
    };
}
pub const DEM_FILENAME0: &str = concat!("demands-v", demand_version!(), ".00.bin");
pub const DEM_FILENAME1: &str = concat!("demands-v", demand_version!(), ".01.bin");
