By Cathay Express, Star Alliance and other contributors.

Last major revision: 7 Jul 2021, last updated: 22 Aug 2024

If you have any suggestions or error corrections, open an
[issue](https://github.com/abc8747/am4/issues) or contact me via our
[Discord server](https://discord.gg/4tVQHtf).

Unless specified or otherwise, all formulae are for <span class="easy">easy</span> mode.

## Ticket Prices
Found: 2019 (AM4 community)

API: [utils.ticket][]

Confidence: <span class="c-good">100%</span> ($R^2 = 1$)

### Pax
#### <span class="easy">Easy</span>
$$
\begin{align*}
\$_\text{Y} &= 0.4d+170 \\
\$_\text{J} &= 0.8d+560 \\
\$_\text{F} &= 1.2d+1200 \\
\end{align*}
$$

where:

- $\$_\text{\{Y,J,F\}}$ are the **autoprice** prices. For *optimal* prices, multiply $\$_\text{Y}$ by 1.1, $\$_\text{J}$ by 1.08, and $\$_\text{F}$ by 1.06.
- $d$: total distance[^1] of the flight.


#### <span class="realism">Realism</span>
$$
\begin{align*}
\$_\text{Y} &= 0.3d+150 \\
\$_\text{J} &= 0.6d+500 \\
\$_\text{F} &= 0.9d+1000 \\
\end{align*}
$$

where:

- $\$_\text{\{Y,J,F\}}$ are the **autoprice** prices. For *optimal* prices, multiply $\$_\text{Y}$ by 1.1, $\$_\text{J}$ by 1.08, and $\$_\text{F}$ by 1.06.
- $d$: total distance[^1] of the flight.

### VIP
#### <span class="easy">Easy</span>

Found: 22 Jul 2021 (Cathay Express)

$$
\begin{align*}
\$_\text{Y} &= 1.7489(0.4d+170) \\
\$_\text{J} &= 1.7489(0.8d+560) \\
\$_\text{F} &= 1.7489(1.2d+1200) \\
\end{align*}
$$

where:

- $\$_\text{\{Y,J,F\}}$ are the **autoprice** prices. For *optimal* prices, multiply $\$_\text{Y}$ by 1.22, $\$_\text{J}$ by 1.195, and $\$_\text{F}$ by 1.175.
- $d$: total distance[^1] of the flight.

#### <span class="realism">Realism</span>
$$
\begin{align*}
\$_\text{Y} &= 1.7489(0.3d+150) \\
\$_\text{J} &= 1.7489(0.6d+500) \\
\$_\text{F} &= 1.7489(0.9d+1000) \\
\end{align*}
$$

where:

- $\$_\text{\{Y,J,F\}}$ are the **autoprice** prices. For *optimal* prices, multiply $\$_\text{Y}$ by 1.22, $\$_\text{J}$ by 1.195, and $\$_\text{F}$ by 1.175.
- $d$: total distance[^1] of the flight.

### Cargo

Found: 10 Mar 2020 (Cathay Express)

#### <span class="easy">Easy</span>

$$
\begin{align*}
\$_\text{L} &= 0.0948283724581252d + 85.2045432642377000 \\
\$_\text{H} &= 0.0689663577640275d + 28.2981124272893000 \\
\end{align*}
$$

where:

- $\$_\text{\{L,H\}}$ are the **autoprice** prices. For *optimal* prices, multiply $\$_\text{L}$ by 1.1 and $\$_\text{H}$ by 1.08.
- $d$: total distance[^1] of the flight.

#### <span class="realism">Realism</span>

$$
\begin{align*}
\$_\text{L} &= 0.0776321822039374d + 85.0567600367807000 \\
\$_\text{H} &= 0.0517742799409248d + 24.6369915396414000 \\
\end{align*}
$$

where:

- $\$_\text{\{L,H\}}$ are the **autoprice** prices. For *optimal* prices, multiply $\$_\text{L}$ by 1.1 and $\$_\text{H}$ by 1.08.
- $d$: total distance[^1] of the flight.

___

## Alliance

### Contribution

Confidence: <span class="c-good">100%</span> ($R^2 = 1, N = 252$)

Found: 11 Apr 2021 (Cathay Express and Star Alliance contributors)

Updated: 13 Apr 2021

API: [utils.route.AircraftRoute.contribution][]

??? warning "SPOILERS: Click to reveal"

    $$\$_\text{C} = \min\left(k_\text{gm}kd\left(3 - \frac{\text{CI}}{100}\right),152 \right) \pm 16\%$$

    where:

    - $d$ is the **direct** distance[^2] of the route
    - $k_\text{gm}$ is a multiplier based on game mode:
        
        $$
        k_\text{gm} = \begin{cases}
            1.5 & \text{if realism} \\
            1 & \text{if easy}
        \end{cases}
        $$

    - $k$ is a function of $d$:

        $$
        k = \begin{cases}
            0.0064 & \text{if } d < 6000 \\
            0.0032 & \text{if } 6000 < d < 10000 \\
            0.0048 & \text{if } d > 10000 \\        
        \end{cases}
        $$

    - $\text{CI}$: between 0 and 200

    Contribution is reduced when the flight distance is less than 1000km. ([REDUCED_CONTRIBUTION warning][utils.route.AircraftRoute.Warning.REDUCED_CONTRIBUTION]
    in [utils.route.AircraftRoute.warnings][])

    !!! note
        From 18 May 2021 - 7 Jul 2021, the contribution was momentarily reduced.

        Updated: 19 May 2021

        $$
        \$_{\text{C,new}} = 0.9013881067562953\$_\text{C}
        $$

        Confidence: <span class="c-moderate">80%</span>

<!-- #### Optimal pure-contribution strategy

Developed: 14 Apr 2021 (Cathay Express)

Check out the guides for a detailed introduction on the concepts required to derive this!

??? warning "SPOILERS: Click to reveal"

    ⚠️ Globally optimal iff $d<6000$!

    **Input**:
    
    - original aircraft speed $v$
    - target flight time after CI $T'$

    **Output**:
    
    - optimal route distance $d$
    - optimal CI
    - contribution per hour $\frac{\$_{\text{CI}}}{T'}$

    **Algorithm**:

    - choose aircraft with largest $v$ to ensure global optimality.
    - $d = \frac{27vT'}{40}$
    - `assert` $d<6000$, reduce $T'$ or $v$ otherwise.
    - $\text{CI} = \frac{2000d}{7vT'}-\frac{600}{7}$
    - $\frac{\$_{\text{CI}}}{T'}=\min(\frac{729}{560}k_\text{gm}kv, \frac{102.6v}{d})$ -->

### Season

=== "Latest, V3 (22 Jul 2021-)"

    Found: 22 Jul 2021 (Cathay Express and Star Alliance contributors)

    Confidence: <span class="c-poor">10%</span>

    Everything we know about seasons:

    1. Individual departures on average yields a higher season contribution.
    2. Recalling aircraft will not remove season contribution.
    3. Season value = total season contribution / 40,000 (exact: /40031.83152)
    4. A greater variety of planes/plane types will contribute more.

    For individual departures:

    - Season contribution do not depend on the ticket price (12 trials), distance (4 trials) or CI (2 trials).
    - Only routes of distance >500km contribute (4 trials).
    - Flights carrying <=7 pax will not contribute (4 trials).
    - Season contribution is not randomised (28 trials)
    - Season contribution is randomised by 10-19%. (4 trials, possibly erroneous)
    - Season contribution is not affected by the aircraft's acheck hours remaining (A320VIP - 3331h to 1771 hours, 31 trials).
    - Season contribution is likely to be proportional to the aircraft's factory acheck hours (needs verification).
    For multiple departures:
    - People with a greater amount of unique planes/plane types will contribute more.

=== "V2 (7 Jul 2021)"

    Found: 8 Jul 2021 (Cathay Express)
    
    Confidence: <span class="c-good">100%</span> ($R^2 = 1$)

    Applicable for both <span class="easy">easy</span> and <span class="realism">realism</span>.

    $$S \approx (\frac{1}{0.00449838V + 0.229888} + 0.317077)V \\$$

=== "V1 (16 Jul 2020)"

    Found: 16 Jul 2020 (Cathay Express)

    Confidence: <span class="c-good">100%</span>

    Applicable for both <span class="easy">easy</span> and <span class="realism">realism</span>.

    Season is the change in total [alliance contribution](#contribution) from the start of Wednesdays, 10:00UTC.

___


## Stock

### SV change

Confidence: <span class="c-good">100%</span> ($R^2 = 1$)

Found: < 20 Aug 2020

Updated: 7 Jul 2021

Each *action* in the game is associated with a change in share value. Applicable for both <span class="easy">easy</span> and <span class="realism">realism</span>.

$$\Delta\$_{\text{S}} = \frac{\$}{k}$$

where:

- $\Delta\$_{\text{S}}$: Change in share value
- $\$$: Amount of money involved during an action
- $k$: some constant based on the action taken:
    - $k = 66666666$ when purchasing an aircraft
    - $k = 40000000$ for *positive* actions:
        - Income earned on departure
        - Cost of route creation
        - Cost of constructing lounges
    - $k = -40000000$ for *negative* actions:
        - Cost of purchasing fuel and $\text{CO}_{2}$
        - Cost of route fee on rerouting, including ferry flight
        - Cost of maintenance or repair
        - Income earned by selling or recalling aircraft
        - Cost of marketing, staff salary, purchasing new hubs, transferring money between banks

No change: hangar unlocking or expansion

Not tested: maintaining lounges

___

## Route

### Distance

Confidence: <span class="c-good">100%</span>

Found: 2019 (Cathay Express and AM4 community)

API: [utils.route.Route.direct_distance][] and `utils.route.Route.calc_distance`

Internally, AM4 calculates the distance between two airports with the [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula):

$$
d = 12742 \arcsin\left(\sqrt{\sin^2\left(\frac{\phi_2-\phi_1}{2}\right) + \cos(\phi_1) \cdot \cos(\phi_2) \cdot \sin^2\left(\frac{\lambda_2-\lambda_1}{2}\right)}\right)
$$

where:

- 12742: Earth's diameter in km
- $\phi_1, \phi_2$: latitude of airport 1 and 2 in radians
- $\lambda_1, \lambda_2$: longitude of airport 1 and 2 in radians

!!! note
    Confusingly, the research panel uses the [spherical law of cosines](https://en.wikipedia.org/wiki/Spherical_law_of_cosines) instead:

    $$
    d = 6371 \arccos\left(\sin(\phi_1)\sin(\phi_2) + \cos(\phi_1)\cos(\phi_2)\cos(\lambda_2 - \lambda_1)\right)
    $$

    While the difference between these two formulae is negligible, all internal calculations rely on the Haversine formula. Therefore, do not use distances from the research panel for ticket price calculations, as **it could lead to empty flights**.

### Demand

Confidence: <span class="c-moderate">90%</span> ($R^2 = 0.985, 0.999$)

=== "Attempt 2"

    A new effort is launched to find the demand formula.

    See [guides](./guides/demand.md).

=== "Attempt 1"

    Found: 1 June 2023 (Cathay Express)

    API Reference: [utils.demand.PaxDemand][], [utils.demand.CargoDemand][]

    The demand is a uniformly distributed random variable:

    $$
    \mathbb{E}(D_{i,j}) = \begin{cases}
        0.0033164(k_i+k_j) + 119.41009 & \text{if capital} \\
        0.0016346(k_i+k_j) + 243.24880 & \text{otherwise}
    \end{cases}
    $$

    where:

    - $\mathbb{E}(D_{i, j})$: expected demand of the route from airport $i$ to airport $j$, that is, the mean demand of routes with the same total tier value.
    - $k$: tier value of the airport. See the [hub cost](#hub-cost) section for more details.


    !!! note
        The true demand of the route is calculated by **adding a random offset** to the mean demand.
        This random offset is uniformly distributed and cannot be meaningfully predicted without knowing the seed of the PRNG.

        Psuedocode: `offset = PRNG(seed=hash(airport1, airport2)).next()` and `hash` is commutative.

        See [GitHub](https://github.com/abc8747/am4/tree/master/docs/assets/scripts/demand-research/old) for attempts to uncover it.

        In practice, we store this in a giant hashtable.


### Marketing Campaign
Applicable for both <span class="easy">easy</span> and <span class="realism">realism</span>.

$$
\begin{align*}
C_{1} &= (2625T + 4500)n_p \\
C_{2} &= (3937.5T + 6750)n_p \\
C_{3} &= (4987.5T + 8550)n_p \\
C_{4} &= (6037.5T + 10350)n_p \\
C_{\text{eco}} &= 8270n_p \\
\end{align*}
$$

where:

- $C_{k}$: Cost of marketing campaign *k*
- $T$: Campaign length, hours
- $n_p$: Total amount of planes ever owned[^3]

### Pax Carried

Confidence: <span class="c-moderate">80%</span> ($N \approx 2000$)

Found: 2021 (Cathay Express)

API: [utils.route.AircraftRoute.estimate_load][]

When there is demand remaining, the load is a random variable with the expected value:

$$
\mathbb{E}(\text{load}) \approx \begin{cases} 
0.0085855R, & \text{if } \alpha > 1 \land s \\
0.0090435R, & \text{if } \alpha > 1 \land \lnot s \\
1 + \alpha(0.0090312R - 1), & \text{if } \alpha \leq 1 \land s \\
1 + \alpha(0.0095265R - 1), & \text{if } \alpha \leq 1 \land \lnot s \\
\end{cases}
$$

where:

- $R$: reputation
- $\alpha$: autoprice ratio: the ratio of the ticket price to the autoprice ticket price (e.g. 1.1):
    - $\alpha > 1$: normally distributed, with standard deviation at 6.8% the mean (heteroscedastic).
    - $\alpha \leq 1$: uniformly distributed, with half-range at 5.2% the mean (heteroscedastic).
- $s$: boolean variable of whether the route has a stopover

When there is insufficient demand, the load is deterministic:

Confidence: <span class="c-moderate">60%</span> (tested at 39% reputation)

$$
\text{load} = \begin{cases}
0.11136R & \text{if }D \geq \text{C} \land \lnot s \\
0.10688R & \text{if }D \geq \text{C} \land s \\
0.3312\frac{D}{C} + 0.1014 & \text{if }D \leq \text{C} \land \lnot s \\
0.3105\frac{D}{C} + 0.1038 & \text{if }D \leq \text{C} \land s \\
\end{cases}
$$

When the demand is less than the intersection between the formula above and y=x, the load is 100% to make sure the demand is depleted:

$$
\text{load} = \begin{cases}
1 & \text{if }D < C\cdot\text{load} \\
\text{load} & \text{otherwise}
\end{cases}
$$

where:

- $D$: demand left before departure
- $C$: capacity of the aircraft
___

### CI

Found: 2019 (AM4 Community)

$$
v = u(0.0035\cdot \text{CI} + 0.3)
$$

where:

- $v$: new speed
- $u$: original speed
- $\text{CI}$: CI, 0-200.

Alternative Expression (useful for optimisation):

$$
\text{CI} = \frac{2000d}{7uT'}-\frac{600}{7}
$$

where:

- $d$: route distance
- $u$: original speed
- $T'$: target flight time after applying CI

### Fuel Consumption

Found: 20 Dec 2020 (Cathay Express)

Confidence: <span class="c-good">100%</span> ($R^2 = 1$)

$$
\text{fuel} = (1 - t_f) \cdot \text{ceil}(d, 2) \cdot c_f \cdot \left(\frac{\text{CI}}{500} + 0.6\right)
$$

where:

- $t_f$: fuel training amount (0-3)
- $\text{ceil}(d, 2)$: total distance[^1], rounded up to 2 decimal places
- $c_f$: fuel consumption of the aircraft. If modified with fuel efficiency, multiply by 0.9.
- $CI$: cost index (0-200)

### CO₂ Consumption

Found: 22 Dec 2020 (Cathay Express)

Confidence: <span class="c-good">100%</span> ($R^2 = 1$)

#### Pax

$$
\text{CO}_2 = (1 - \frac{t_c}{100}) \cdot \text{ceil}(d, 2) \cdot c_c \cdot \left(y_l + 2j_l + 3f_l + y_c + j_c + f_c\right) \cdot \left(\frac{\text{CI}}{2000} + 0.9\right)
$$

where:

- $t_c$: CO₂ training amount (0-5)
- $\text{ceil}(d, 2)$: total distance[^1], rounded up to 2 decimal places
- $c_c$: CO₂ consumption of the aircraft. If modified with CO₂ efficiency, multiply by 0.9.
- $y_l$, $j_l$, $f_l$ are the number of economy, business, and first class passenger loaded
- $y_c$, $j_c$, $f_c$ are the number of economy, business, and first class seats
- $CI$: cost index (0-200)

#### Cargo

$$
\text{CO}_2 = (1 - \frac{t_c}{100}) \cdot \text{ceil}(d, 2) \cdot c_c \cdot \left(0.7l_l  + h_l + 0.7l_c  + h_c\right) \cdot C \cdot \left(\frac{\text{CI}}{2000} + 0.9\right)
$$

where:

- $t_c$: CO₂ training amount (0-5)
- $\text{ceil}(d, 2)$: total distance[^1], rounded up to 2 decimal places
- $c_c$: CO₂ consumption of the aircraft. If modified with CO₂ efficiency, multiply by 0.9.
- $l_l$, $h_l$: actual load of large and heavy cargo (%)
- $l_c$, $h_c$: configuration of large and heavy cargo (%)
- $C$: the total equivalent cargo capacity of the aircraft (lbs)
- $CI$: cost index (0-200)

### Creation cost

Found: 2024 (Alliance *Ariving Germany*)

Confidence: <span class="c-good">100%</span>

#### Pax, <span class="easy">Easy</span>

$$
C = 0.4 (d + (y_c \cdot \lfloor 0.4d + 170 \rfloor) + (j_c \cdot \lfloor 0.8d + 560 \rfloor) + (f_c \cdot \lfloor 1.2d + 1200 \rfloor))
$$

where:

- $d$: total distance[^1]
- $y_c$, $j_c$, $f_c$ are the number of economy, business, and first class seats

#### Pax, <span class="realism">Realism</span>

$$
C = (d \cdot n_p) + (y_c \cdot \lfloor 0.3d + 150 \rfloor) + (j_c \cdot \lfloor 0.6d + 500 \rfloor) + (f_c \cdot \lfloor 0.9d + 1000 \rfloor)
$$

where:

- $d$: total distance[^1]
- $n_p$: total amount of planes ever owned[^3]
- $y_c$, $j_c$, $f_c$ are the number of economy, business, and first class seats

### Best Configuration

Developed: 2019 (AM4 Community, Cathay Express)

See [guides/configuration](guides/configuration.md)

**Input:**

- Seat classes: $S = \{s_1, s_2, \ldots, s_n\}$
- Units of capacity: $c_1, c_2, \ldots, c_n$
- Demand: $d_1, d_2, \ldots, d_n$
- Flights per day: $f$
- Capacity: $C$

**Output**:

- Optimal seat configuration: $\text{seats} = (\text{seats}_1, \text{seats}_2, \ldots, \text{seats}_n)$

**Algorithm:**

1. $C_{\text{remaining}} \leftarrow C$.
2. **for** $i \leftarrow 1$ **to** $n$:
    - $\text{seats}_i \leftarrow \left\lfloor\frac{d_i}{f}\right\rfloor$.
    - **if** $\text{seats}_i \cdot c_i \leq C_{\text{remaining}}$:
        - $C_{\text{remaining}} \leftarrow C_{\text{remaining}} - \text{seats}_i \cdot c_i$.
    - **else**:
        - $\text{seats}_i \leftarrow \left\lfloor\frac{C_{\text{remaining}}}{c_i}\right\rfloor$.
        - $C_{\text{remaining}} \leftarrow 0$.
3. **return** $\text{seats} = (\text{seats}_1, \text{seats}_2, \ldots, \text{seats}_n)$.

#### Pax seat ordering

Assuming the use of 1.1x, 1.08x, and 1.06x multipliers from the autoprice ticket prices.

| Game Mode                                         | Distance Range (km) | Best Seat Class Order |
| ------------------------------------------------- | ------------------- | --------------------- |
| <span class="easy">Easy</span> {:rowspan=4}       | < 14425             | F>J>Y                 |
| 14425 - 14812.5                                   | F>Y>J               |
| 14812.5 - 15200                                   | Y>F>J               |
| > 15200                                           | Y>J>F               |
| <span class="realism">Realism</span> {:rowspan=4} | < 13888.89          | F>J>Y                 |
| 13888.89 - 15694.44                               | J>F>Y               |
| 15694.44 - 17500                                  | J>Y>F               |
| > 17500                                           | Y>J>F               |

#### Cargo seat ordering

Assuming the use of 1.1x and 1.08x multipliers from the autoprice ticket prices.

| Game Mode                                   | Distance Range (km) | Best Seat Class Order |
| ------------------------------------------- | ------------------- | --------------------- |
| <span class="easy">Easy</span> {:rowspan=2} | <23908              | L>H                   |
| >23908                                      | H>L                 |
| <span class="realism">Realism</span>        | all                 | L>H                   |


## Airport

### Hub Cost

Found: Mar 2020 (Cathay Express)

Confidence: <span class="c-good">100%</span> ($R^2 = 1$)

API: [utils.airport.Airport.hub_cost][]

Applicable for both <span class="easy">easy</span> and <span class="realism">realism</span>.


$$C_{k} = kp + k$$

where:

- $C_{k}$: Cost of hub with tier value *k* ($k$ is hardcoded)
- $p$: Total amount of planes ever owned

The maximum tier value is a capital route.

### Lounge

#### Construction time

Found: 7 Jul 2021 (Cathay Express)

Confidence: <span class="c-good">100%</span> ($R^2 = 1$)

Applicable for both <span class="easy">easy</span> and <span class="realism">realism</span>.

$$
\begin{align*}
T_{\text{regular}} &= 500n_l + 2000 \\
T_{\text{highend}} &= 700n_l + 3600 \\
T_{\text{exclusive}} &= 900n_l + 7200 \\
\end{align*}
$$

where:

- $T$: time needed to construct the lounge
- $n_l$: number of lounges owned

#### Construction cost

Found: 07 Jul 2021

Confidence: <span class="c-moderate">75%</span> (Lounge fee is lower for airlines with less cash?)

Applicable for both <span class="easy">easy</span> and <span class="realism">realism</span>.

$$
\begin{align*}
C_{\text{regular}} &= 1500000n_l + 37500000 \\
C_{\text{highend}} &= 5000000n_l + 125000000 \\
C_{\text{exclusive}} &= 15000000n_l + 375000000 \\
\end{align*}
$$

where:

- $C$: cost needed to construct the lounge
- $n_l$: number of lounges owned

### Hangar expansion

Found*: 21 Jan 2022 (Cathay Express)

Updated: 25 Jan 2022 (Cathay Express, reduced by \~50%)

See [GitHub](https://github.com/abc8747/am4/tree/master/docs/assets/scripts/hangar-research/main.py) for raw data.

![Hangar Expansion](./assets/img/hangar-10.svg){: .center}

Confidence: <span class="c-poor">30%</span> ($R^2 = 0.94$)

$$C_{10} \approx 147503 \cdot {1.04839}^{x}$$

where:

- $C_{10}$: cost needed to add 10 planes
- $x$: fleet limit

. The cost needed to add 1 plane is:

$$C_{1} = \left\lfloor \frac{1}{10}C_{10} \right\rfloor$$

[^1]: A flight A→B→C has total distance is `distance(A, B) + distance(B, C)`.
[^2]: A flight A→B→C has direct distance is `distance(A, C)`.
[^3]: The total number of planes ever purchased can be found by navigating to the aircraft ordering page and checking the prefix of the registration.

___

## Aircraft
### Wear

Found: 2021 (Cathay Express)

Confidence: <span class="c-good">100%</span> ($R^2 = 1$). Applicable for both <span class="easy">easy</span> and <span class="realism">realism</span>.

$$
\text{wear} \sim \mathcal{U}(0, 0.015\cdot(1-0.02t_r))
$$

where:

- $t_r$: repair training amount (0-5)
- $\mathcal{U}$: uniform distribution

Equivalently, the expected wear is 0.75% per departure, which decreases by 2% per training point.

### Repair Cost

Found: 2020 (Cathay Express)

Confidence: <span class="c-good">100%</span> ($R^2 = 1$). Untested on realism.

$$
C_r = 0.001C(1 - 0.02t_r)\cdot\text{wear}
$$

where:

- $C$: aircraft [cost][utils.aircraft.Aircraft.cost]
- $t_r$: repair training amount (0-5)
- $\text{wear}$: [aircraft wear](#wear) percentage, 0-1 

### Repair Time

Found: 29 June 2024 (Cathay Express, Point Connect)

Confidence: <span class="c-good">99%</span> ($R^2 = .995$). Applicable for both <span class="easy">easy</span> and <span class="realism">realism</span>.

$$
T = 480000 \cdot \text{wear}+3600
$$

where:

- $T$: downtime (seconds)
- $\text{wear}$: [aircraft wear](#wear) percentage, 0-1 

### Check Time

Found: 29 June 2024 (Cathay Express, Point Connect)
Updated: 9 July 2024

Confidence: <span class="c-good">95%</span>.
Note that the check cost for easy players are half of realism.

#### Easy

$$
T = 0.01 \cdot C+1860
$$

#### Realism

$$
T = 0.01 \cdot C+3700
$$

where:

- $T$: downtime (seconds)
- $C$: [check cost][utils.aircraft.Aircraft.check_cost]

___

## Miscellaneous

### Level Bar

Found: Mar 4 2020 (Cathay Express)

Confidence: <span class="c-good">100%</span> ($R^2 = 1$). Applicable for both <span class="easy">easy</span> and <span class="realism">realism</span>.


$$f_{l + 1} = 8l + 4$$

where:

- $f_{l+1}$: number of flights needed to reach the next level
- $l$: current level number