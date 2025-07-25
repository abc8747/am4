# ![logo](docs/assets/img/logo-small.png) am4

[![](https://dcbadge.vercel.app/api/server/4tVQHtf?style=flat)](https://discord.gg/4tVQHtf) [![CI](https://github.com/abc8747/am4/actions/workflows/ci.yml/badge.svg)](https://github.com/abc8747/am4/actions/workflows/ci.yml)

[Airline Manager 4](https://airlinemanager.com) is an online multiplayer game whose goal is to build an airline from scratch. Rapid progression in the game requires thorough market research and route planning to high demand destinations, while considering recurring fuel/CO2 costs, and conforming to aircraft range/runway requirement constraints.

The weekly leaderboards and alliances are highly competitive arenas which require extensive knowledge in the game: this repository contains a set of high-performance foundational tools, built for the more technical-oriented players. The project is structured as follows:

- [x] core calculations written in C++ for exhaustive searches ([`src/am4/utils`](./src/am4/utils))
- [x] a Python web API exposing the core ([`src/am4/api`](./src/am4/api/))
- [x] an sqlite database storing user settings ([`src/am4/db`](./src/am4/db/)) 
- [x] a Python discord bot for our community ([`src/am4/bot`](./src/am4/bot/))

A new version (v0.2) is currently under development* so you can run it on the web ([`am4help.com`](https://am4help.com/)) and completely offline.

- [ ] core library in Rust ([`am4`](./am4/))
- [ ] a web frontend using WASM ([`am4-web`](./am4-web/))
- [ ] a discord bot

<div align="center">
  <img src="docs/assets/img/overview.drawio.svg" alt="overview">
</div>

## Installation (v0.1)
For the latest core utils:
```sh
pip install am4
```

For the bot and API, clone the repository and run:
```sh
docker build -t am4 .
docker run -d -p 8002:8002 -p 8090:8090 --name am4-dev am4
```

Please see the [documentation](https://abc8747.github.io/am4/) for detailed formulae, installation and usage guides.