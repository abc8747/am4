!!! important
    The [core](./core.md) must be built before running the bot.

### Configuration File
The database and API can run out of the box without any configuration file using the [defaults](https://github.com/abc8747/am4/tree/master/src/am4/config.py). However, running the discord bot/building the frontend will require you to provide custom config.

To do this, rename `./config.example.json` to `./config.json` and modify it. Then, run:
```sh
python3 -m src.am4 cfg set config.json
```
This will verify and persist the settings to `./src/am4/.config.json`, which is used in future calls.

### Database
User authentication is handled by [pocketbase](https://github.com/pocketbase/pocketbase), a lightweight SQLite-based backend supporting OAuth2.0.

It is automatically started before the API/discord bot when using the CLI.

### Discord bot
It is similar to the API in design, but is more sophisticated in that it reads users' roles to determine their game mode.

To start it, use:
```sh
python3 -m src.am4 start bot
```

??? note "Documentation for legacy code, click to expand."

    ### Features
    Old bot:

    - calculates essential statistics
        - most distance-efficient stopovers
        - route demands, best seat configurations, best ticket prices, estimated income
        - player rank, mode, achievements, fleet
        - alliance rank, share value, contribution
        - aircraft characteristics and profit
        - airport characteristics
    - CSV export for route queries
    - fuel/CO2 notifications
    - aircraft characteristics comparisons
    - internal *Star Alliance* tools (now disbanded)
        - adding competitor alliances to watchlist
        - alliance comparisons over time: value,contribution/day, rate of changes
        - realtime alliance-member comparisons: SV/contribution distribution
        - member tracking: cheat detection tools, departure pattern identification

    ### Public
    - `$route|stop <airport> <airport> <aircraft> [flights_per_day] [reputation]`: finds the best route between two airports
      
      ![route](../assets/img/route.png)

    - `$routes <airport> <aircraft> <max_distance> <flights_per_day> [reputation]`: finds the best destinations from a certain airport, sorted by decreasing estimated income
      
      ![routes](../assets/img/routes.png)

    - `$user [player]`: shows player (and associated alliance if found) statistics
      
      ![user](../assets/img/user.png)

    - `$fleet [player]`: shows player fleet and estimated income
      
      ![fleet](../assets/img/fleet.png)

    - `$info <aircraft>`: shows basic aircraft information and rough profit estimations
      
      ![info](../assets/img/info.png)

    - `$compare <aircraft>`: compares two aircrafts
      
      ![compare](../assets/img/compare.png)

    - `$search <aircraft>`: finds the associated aircraft shortname for aircraft commands
      
      ![search](../assets/img/search.png)

    - `$airport <airport>`: shows airport information
      
      ![airport](../assets/img/airport.png)

    - `$price f[fuel_price] c[co2_price]`: notifies everyone for the fuel price
      
      ![price](../assets/img/price.png)

    ### Internal Alliance Tools
    - `$memberCompare <player> <player>`: compares descending structure of contribution/day and SV
      
      ![member-compare](../assets/img/member-compare.png)
    - `$alliance <alliance>`: shows AV progression and d(AV)/dt.
      
      ![alliance](../assets/img/alliance.png)
    - `$allianceCompare <alliance> <alliance>`: compares AV progression and gap difference over time, shows 48h/12h-average contribution/day graphs
      
      ![alliance-compare](../assets/img/alliance-compare.png)
    - `$member <player> [player[]]` shows contribution/day, total contribution and SV history for 1+ members
      
      ![member](../assets/img/member.png)
    - `$actions <player> [maxResults]`: shows log of estimated departures, contributions and income
      
      ![member-compare](../assets/img/member-compare.png)
    - `$watchlist [add|+, remove|rm|-] [alliance]`: shows, adds or remove alliance(s) to the watchlist