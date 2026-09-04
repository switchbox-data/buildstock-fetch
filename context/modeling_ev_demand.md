# Modeling EV demand on top of 2024 ResStock data

We append electric vehicle (EV) charging profiles to buildings from the 2024 ResStock
release. The goal is to quantify how time-of-use (TOU) rates shift EV charging and how
much EV owners can save on their bills, then (later) compare those bills to the costs
EVs impose on the grid.

Scenario knobs live in YAML (`utils/EVs/configs/md_2024.yaml`). Run:

```bash
python -m utils.EVs.ev_demand --config utils/EVs/configs/md_2024.yaml
```

The Maryland scenario uses a **2018** calendar (`2018-01-01T04:00:00` through
`2019-01-01T03:00:00`) so EV hours align with AMY2018 weather and ResStock base loads
(non-leap; no Feb 29 gap). The window must start at 04:00 and end at 03:00 so it lines
up with the NHTS travel day. Date-only config values are rejected.

## Pipeline overview

`EVDemandCalculator` (`ev_demand.py`) is the orchestrator. For each batch of ResStock
buildings it:

1. Assigns EV slots to households.
2. Matches each slot to weekday/weekend NHTS travel-day templates.
3. Replays those templates across the simulation year (with noise and seam legs).
4. Sizes a battery from peak daily driving duty (temperature-scaled when enabled).
5. Assigns a home-charging energy fraction and a home charger (L1 vs L2).
6. Simulates hourly presence, discharge, charging, and SOC.

Knobs are loaded into `EVDemandConfig` from YAML — scenario values have no Python
defaults. Helpers in `ev_utils.py` load ResStock metadata, NHTS, weather, and the
2025 ResStock EV TSVs; they do not change the model.

The Maryland YAML (`utils/EVs/configs/md_2024.yaml`) sets: `resstock_adoption`,
`temperature_adjustment=resstock`, `home_charging_fraction_assignment=resstock`,
`charger_assignment=resstock`, `charging_strategy=off_peak_immediate`.

---

## `EVAdoptionSampler` / `VehicleOwnershipModel`

**Question:** which ResStock dwellings get an EV, and how many slots?

Chosen by `sampling.ev_assignment`. Vacant units never get EVs. NLR treats households
and dwelling units as synonymous in ACS/PUMS.

### `ev_assignment = resstock_adoption` (Maryland default) — `EVAdoptionSampler`

At most one EV per occupied household. Segment probability $p_i$ comes from
[`Electric_Vehicle_Ownership.tsv`](https://github.com/NatLabRockies/resstock/tree/develop/project_national/housing_characteristics)
(2025 ResStock): $P(\mathrm{EV})$ by FPL, building type, tenure, and PUMA, applied to
the 2024 building sample.

How NREL built that lookup (EV Integration Report,
https://www.osti.gov/biblio/2584243; Technical Reference Guide,
https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2025/resstock_amy2018_release_1/ResStockTechnicalReferenceGuide_2025_1.pdf):

1. 2023 Experian registrations × 2016 ACS vehicles per household → county EV-per-household.
2. Map counties to PUMAs (tract-weighted; rates constant within county) and join 2019 5-year PUMS.
3. Scale EIA RECS 2020 $P(\mathrm{EV} \mid \mathrm{FPL}, \text{building type}, \text{tenure})$ so that
   $P(\mathrm{EV} \mid \mathrm{PUMA}) = \sum P(\mathrm{EV} \mid \mathrm{PUMA}, \mathrm{FPL}, \ldots)\, P(\mathrm{FPL}, \ldots \mid \mathrm{PUMA})$
   via a PUMA-specific factor $k_{\mathrm{PUMA}}$.

**Baseline draw.** Independent Bernoulli: occupied building $i$ gets an EV iff
$u_i \lt p_i$, $u_i \sim U(0,1)$. That differs from ResStock 2025 quota sampling,
which can under-assign when segment expectations are fractional (e.g. MD 2025: 165
expected EVs vs 127 assigned).

**Targeted stock share.** Optional `sampling.target_adoption_rate = τ` ∈ [0, 1]
priority-samples occupied buildings on key $u_i / p_i$ and takes a ResStock-weight
prefix until occupied housing-stock EV share equals τ. Segment *rates* stay proportional
to $p_i$; $p=0$ buildings adopt last. Same `random_state` + metadata row order ⇒
nested EV sets as τ rises. Higher τ does not require rebuilding the TSV; rebuilding
from new PUMA/county registrations is only needed if *relative* segment rates should
change.

### `ev_assignment = pums_vehicles` — `VehicleOwnershipModel`

Fit a multinomial logistic regression on PUMS (occupants, income, metro, household
weights) to predict household vehicle count $0 \ldots$ `max_vehicles` (counts above
the cap are clipped). Treat every predicted vehicle as an EV. NHTS matching then uses
vehicle count (`match_on_vehicles=True`). `target_adoption_rate` is ignored.

---

## `NHTSProfileSampler` (and `nhts_tours`)

**Question:** given an EV slot, what does a typical weekday and weekend look like?

2022 NHTS travel days run 4am → 3:59am the next day
(https://nhts.ornl.gov/media/2022/doc/2022%20NextGen%20NHTS%20User's%20Guide%20V201_PubUse.pdf).
`load_nhts_data` keeps the state's **census division**. The match pool is built from
vehicle-owning households and their **owned** light-duty inventory (cars, vans, SUVs,
pickups), not from the trip file alone. That way a car that stayed in the driveway on
the household's survey day can still be a template (home all day, 0 miles). Those idle
days are optional — Maryland currently drops them
(`include_zero_driving_days_in_match_pool: false`). 

Each NHTS vehicle-day becomes a `TripProfile`:

- **Legs** (drive intervals + miles) — used later for discharge.
- **Tours** (leave-home → return-home) — used later for presence. A vehicle parked at
  work is away even when not driving. Tours are chained at NHTS minute resolution
  (`STRTTIME` / `ENDTIME`, `WHYFROM` / `WHYTO`), then snapped to clock hours. Home
  purposes are 01 (home) and 02 (work from home). Dwell time is ignored: sitting at
  work for eight hours is still one tour.

**Pool filters**

- Idle inventory vehicles (owned, no trips that survey day) are empty templates when
  `include_zero_driving_days_in_match_pool` is true; Maryland currently excludes them.
- Driven vehicle-days whose daily miles fall outside
  `[nhts_daily_miles_percentile_low, nhts_daily_miles_percentile_high]` are dropped
  (Maryland: 0–100).
- Driven days must **touch home**. Both-ends-open days (away → away under daily
  repeat) are dropped. One-open days stay if they pass the feasibility screen.

**Reference feasibility prefilter.** Before demographic matching, each weekday/weekend
template must be supportable by **at least one** ResStock stock pack on a Level 2
charger (capacity ∩ daily-repeat recharge, with the same buffers used later). Idle
days always pass. Design miles are the **annual peak day**, not the survey day: scaled
by the ResStock discharge curve at `nhts_feasibility_temperature_f` (default 0°F,
×2.26) and padded by $1 + 2 \times$ `miles_noise_std_fraction` for replay noise.

Open templates (start *or* end away from home, e.g. a night-shift day) cannot be
replayed as a closed home-based day. Annual expansion later imputes a synthetic
`leave_home` or `return_home` **seam leg** by mirroring the opposite edge of the
same tour (same miles and duration as the observed outbound or homebound). Those
miles are extra discharge and the drive occupies hours that cannot be used for
home charging, so the feasibility screen sizes the template *as it will be
replayed*, not as NHTS reported it. (Both-ends-open days are already dropped:
under daily repeat they never come home.) Residual battery/charger infeasibility
after annual expansion is a hard error (no post-assignment redraw).

**Matching.** Household-first: draw a demographically similar NHTS household, then one
of its inventory vehicles (may be idle). Weekday and weekend draws are independent
(each NHTS HH has one `TRAVDAY`). Cascade (urban/rural first; prefer occupants over
income when dropping a dimension; vehicle-count tier only in `pums_vehicles`):

1. `urban_income_occupants` (plus vehicles under `pums_vehicles`)
2. `urban_occupants`
3. `urban_income`
4. `income_occupants`
5. `income` only

Income is three bins (≤\$50k / \$50–150k / \$150k+); occupants are 1 / 2 / 3+. If no tier
has enough profiles, matching raises.

Trips are hourly: departure = clock hour of start; arrival = end time if exactly on the
hour, else rounded up. Clock hours 0–3 sit on the next calendar morning of the travel
day. NHTS trip weights only permute sampling order.

---

## `TripScheduleGenerator`

**Question:** turn two day-templates into a year of drive legs and tours.

For each travel day $D$ (4am on $D$ through 3:59am on $D+1$):

- Replay the weekday or weekend template. Missing / idle day type → treat as home (no
  rows; presence stays home).
- Independent hour offsets on each **drive leg**'s departure and arrival from
  $\{-2,-1,0,1,2\}$ with probabilities $(0.05, 0.10, 0.70, 0.10, 0.05)$.
- Miles ~ truncated Normal(logged, $0.1 \times$ logged) on $[0, \infty)$.
- Clip to travel-day hours $[4, 28)$. Pack **tours** so away windows do not overlap
  or spill into the next travel day; mid-tour legs are not forced apart from each
  other. Tours that already end away extend to the travel-day end for presence.

**Synthetic seams.** Open home/away boundaries get imputed `return_home` / `leave_home`
legs (mirrored miles and duration). Duration may clamp to the overnight gap if the
mirror does not fit; miles are kept so battery sizing is not understated. One-open
overnight home uses a **centered self-repeat** seam: home hours =
$0.5 \times \max(0, G - d)$ where $G$ is the overnight gap and $d$ is the
mirrored drive. Placement of the seam in the gap uses midpoint-heavy triangular
weights (no preferred clock hour).

---

## Temperature adjustment (`ev_utils.resstock_temp_power_mult`)

**Question:** how much extra energy does a cold (or hot) hour cost?

When `temperature.temperature_adjustment=resstock` (Maryland default), driving
discharge is Autonomie kWh/mi × `power_mult(T)`, the OpenStudio-HPXML / Geotab–Recurrent
curve used in ResStock. Outdoor dry-bulb comes from the building's ResStock weather
station CSV. $T$ is clipped to 0–100°F; 0°F is the worst case (×2.26).

`ChargingSimulator.build_hourly_temp_scaled_miles` expands trip miles onto drive hours
and applies that multiplier **once**. The same hourly duty array is reused for battery
sizing and later SOC discharge (`discharge = miles × kwh_per_mile × fraction_charged_home`).
When adjustment is `none`, the multiplier is 1.

Assumption: charger power is **not** temperature-dependent; only driving kWh is.

---

## `EVBatteryAssigner`

**Question:** which pack (usable kWh and kWh/mi) does this vehicle get?

Draw from the national ResStock 2025 BEV stock (Experian/TEMPO option shares + Autonomie
usable kWh and kWh/mi), **restricted** to options that:

1. Cover peak daily **duty miles** × kWh/mi × `(1 + capacity_buffer)` (Maryland buffer
   0.2).
2. Can refill that buffered energy on Level 2 during home hours on **that same peak
   travel day** (daily-repeat energy balance).

Duty miles are temperature-scaled when temp adjustment is on. Sizing uses **full** trip
duty, not the home-charging fraction — the pack must cover physical driving; away
charging is handled later by shrinking residential discharge.

Probabilities are renormalized within the feasible set. If no option works, assignment
raises (the NHTS screen is supposed to have already dropped hopeless templates).

---

## `EVHomeChargingFractionAssigner`

**Question:** what share of trip energy is attributed to the residential meter?

`home_charging.home_charging_fraction_assignment`:

- `none` → `fraction_charged_home = 1.0` (all trip energy on the home meter).
- `resstock` (Maryland default) — multinomial sample of EIA 2020 RECS bins from
  `Electric_Vehicle_Charge_At_Home.tsv` (FPL × building type; Speake et al. 2025 §3.5),
  mapped to midpoints (0–19% → 0.10, …, 100% → 1.00).

This scalar multiplies **residential** discharge and charger SOC feasibility only. Away
charging is excluded by shrinking home discharge, not by modeling workplace/public
charge events.

---

## `EVChargerAssigner`

**Question:** Level 1 or Level 2 at home, and at what kW?

- `charging.charger_assignment=fixed` — every EV uses `charger_power_kw` (default 7.2).
- `resstock` (Maryland default) — sample L1 vs L2 from `Electric_Vehicle_Charger.tsv`
  (RECS 2020 `EVCHRGTYPE`; FPL × building type × tenure; Speake et al. 2025 §3.2)
  among levels that cover home-attributed discharge under perfect-foresight
  **immediate** charging, with discharge inflated by `charger_buffer_fraction` (0.2).
  If only one level is feasible, assign that one. If neither is, raise.

Default powers: L1 = 1.6 kW, L2 = 7.2 kW (typical 32 A / 240 V nameplate). ResStock's
TRG maps L2 to 5.69 kW (average observed 240 V draw); override via YAML if you want
that.

Assumption: every simulated slot already owns an EV, so we join only the TSV's
"ownership = Yes" rows. Units without an EV are not in this pipeline.

---

## `ChargingSimulator` and `charging.py`

**Question:** when is the vehicle home, how much energy does it draw, and when does it
charge?

The pack is lossless. It starts full unless `initial_soc_kwh` is set. Within each hour,
discharge is applied first, then charge. Presence uses **tours** (`at_home`); discharge
uses **drive legs** only.

Residential discharge in hour $t$:

$$
\text{duty miles}_t \times \text{kWh/mi} \times \texttt{fraction\_charged\_home}
$$

(with temperature scaling already folded into duty miles). Trip energy the pack cannot
cover is treated as public charging: SOC is clamped to empty and `soc_underflow` is
flagged. There is no explicit public-charging session.

Maryland default strategy is `off_peak_immediate`. The four policies:

1. **Immediate.** Charge at assigned home power whenever home and not full.

2. **Cost-minimizing.** Year-long perfect-foresight LP: minimize
   $\sum_t p_t x^{CB}_t + \mathrm{penalty}\, x^{SL}_t$ subject to SOC dynamics,
   $0 \le x^{CB}_t \le C^B_i$ when home (else 0), $0 \le s_t \le K^B_i$. $C^B_i$
   and $K^B_i$ are per vehicle. Maryland prices are BGE Schedule EV **summer SOS**
   on/off rates applied year-round (`tou_*_price_*`; supply only, no delivery). Shed
   penalty defaults to $10^6$ \$/kWh so shedding is only for LP feasibility.
   $T = 8{,}760$ hours for the 2018 non-leap window. No terminal constraint
   $s_T = s_0$.

3. **Off-peak (`off_peak`).** Charge only off-peak while home until a daily
   $\mathrm{SOC}^{req}$ (comfort floor + day's discharge + safety buffer). Adapted
   from https://github.com/switchbox-data/rate-design-platform-archive (daytime
   off-peak charging is allowed). Perfect foresight of that day's trip energy.

4. **TOU Immediate (`off_peak_immediate`).** Charge at assigned power whenever home and
   **off-peak**, filling toward a full pack (not $\mathrm{SOC}^{req}$). With
   `allow_emergency_peak_charging: true`, on-peak home hours may charge if a **48-hour**
   forecast that uses only future off-peak home charging would underflow on any trip.

Maryland TOU windows (BGE Schedule EV **summer** hours, year-round): weekdays 10am–8pm
peak (clock hours 10–19); weekends entirely off-peak; holidays **not** special. DST is
not modeled.

---

## Remaining work

- No terminal SOC constraint $s_T = s_0$. A wraparound constraint would force the
  pack to end the year where it started (needed for a repeating annual cycle). An alternative is to simulate a **burn-in** of extra travel days before
  $t=0$ and a few extra days after the reporting window, then drop those hours
  when writing results: start-of-year SOC is then the post-burn-in state rather
  than an assumed full pack, and end-of-year charging is not truncated by the
  calendar cutoff.
- Off-peak and cost-min still assume perfect foresight of the (relevant) trip schedule.
- Public-charging / underflow hours still need a richer treatment than “flag and empty.”
- Off-peak and cost-min still assume perfect foresight of the (relevant) trip schedule.
- Public-charging / underflow hours still need a richer treatment than “flag and empty.”
