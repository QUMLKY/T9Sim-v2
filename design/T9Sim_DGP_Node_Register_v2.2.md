# T9Sim DGP Node Register (v2.2)

**Forked 22 August 2026 from `T9Sim_DGP_Node_Register.md`, which is v2's trusted record and is not
edited.** The v2 file stays where it is because `t9v2/tools/build_graph.py` reads it by that exact
path: `graph.yaml` is generated from the register and the frozen column order is parsed out of the
Specification. Repointing those two path constants at these v2.2 files is a **step 5** action, taken
in the same change as the generator law. Until then `main`, tag `v2.0.0` and tag `v2-corrected-1M`
all still build from the v2 files and reproduce byte-identically.

## Change log against v2

*One row per design unit. Numbered `U1…Un`, never `D1…Dn`, because the `D` family is already the
Specification's time columns (timestamp, hour_of_day, day_of_week, week).*

| Unit | Change |
|---|---|
| U1 | **H5 `min_winning_price`**, law `max(LU7, H1)`, on every row, `C1 none / C2 none / C3 all / C4 all`. Tier-2 price-head target, never a per-row feature. Reinstates v1's arm B1 in an all-rows form. Applied 22 Aug 2026 from a verified 44-site sweep; see `docs/T9Sim_Rebuild_Plan_v2.2.md`, build order step 3 |

---

**13 August 2026.** Every node of the generator, one row each, with its type, parents, law, calibration source, constraint and v2 decision. This is the build input for `config/graph.yaml`.

The type codes T1 to T4 are defined in `docs/T9Sim_Specification_v2.md`, along with the mixed-type rule and the list of nodes carrying a probability table. The open decisions moved there too, into its open-items section.

## Type counts (dead rows excluded)

| Type | Count | Members in one line |
|---|---|---|
| T1 calibrated draw (no parents) | 18 | LU1, A1, LA1, adv_tier, C4, LC1, LC2, LR1–LR6, B3, B6, D4, floor_shape, pay_shape |
| T2 probability table (CPT) | 9 | D2, pair_idx, c_idx, A3, A5, D3 (raked); A2, A4, _size (nested) |
| T3 conditional distribution | 17 | LU2–LU6, LA2, u_rows, D1, participate_k, b_k, H2, E1, E2, F1, F2, G1, G2 |
| T4 deterministic formula | 35 | user_id, B1, B2, B4, B5, C1, C2, C3, sample_weight, app_i, user_vbin, the 7 truth mediators, the 5 estimands, z, H1, base_e, eltv_b2, LU7, H9, H5, H3, sold_lost, H4, G3, G4 |
| **Total active nodes** | **79** | nothing is excluded; the 3 former CPT candidates were wired at O1 and sit in the T2 row above |

---

## 3. The node register

One register, ordered by generation stage. Step vocabulary: `pool-users`, `pool-apps`, `pool-campaigns`, `pool-rivals`, then per auction `1 join`, `2 context`, `3 truth`, `4 floor`, `5 participation`, `6 rival bids`, `7 our bid`, `8 win`, `9 settle`, `10 funnel` (matching spec §2.1, `docs/v1/T9Sim_Specification_v10.md:238-247`). Column "—" means generator-internal, never a parquet column (`emitted: false`). File paths are repo-relative; code paths are under `t9_sim/src/t9sim/`, config under `t9_sim/config/`.

### 3.1 pool-users — the user pool is built once; every auction later inherits one user row

| # | Node | Column | Role | Plate | Stream | Parents | Type | Law (one line) | Source | Constraint | v1 status | v2 decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LU1 | lu1_archetype | latent | user | user_pool | none | T1 | draw from shares 0.02/0.08/0.20/0.40/0.30 over whale, engaged_spender, casual, time_filler, inactive | `market.yaml archetype_shares` 0.02 / 0.08 / 0.20 / 0.40 / 0.30, route `specified` (spec 2.0: taken from this project's own design documents, never quoted to an outside source); v1 copy at `archetypes.yaml:15-20`, whose BM synthesis note cites Newzoo/Unity/Moloco bands and records the shares as not yet quoted | rake target of R2 (pairing rows sum to the archetype mix); R1 preserves the archetype-weighted hour marginal | `user_profiles.py:52` live | keep; the archetype hub, parent of LU2–LU6, of D2, and of the three tables wired at O1 (A3 os, A5 device_type, D3 day_of_week) |
| 2 | A1 | region | observable | user | user_pool | none | T1 | draw from the iPinYou region pmf, anonymised codes | `ipinyou_region_distribution.csv` | R10 renorm; archetype tilt dropped 17 Jun (`bn_cpts.yaml:9-10`) | `user_profiles.py:36,103` live | keep; drop decision settled, do not reopen |
| 3 | A2 | city | observable | user | user_pool | A1 | T2 nested | draw from cities[region] pmf | `ipinyou_city_distribution.csv`, load `config_loader.py:124-130` | R10 per region | `user_profiles.py:38-41` live | keep |
| 4 | A3 | os | observable | user | user_pool | LU1 | T2 | draw from the archetype's os row; the table is raked so the 0.59/0.41 marginal survives (O1, wired 15 Aug 2026) | `device.os_split` 0.59 / 0.41, route `inferred` (O3 3a, decided 15 Aug 2026: never externally checked, and will not be) | IPF-raked to the `device.os_split` marginal 0.59 iOS / 0.41 Android (tol 1e-6, `tables.yaml` A3_os_given_archetype), so wiring the LU1 tilt at O1 leaves the population os mix unmoved | `user_profiles.py:45-50` live; its CPT is dead | keep structure; no external sourcing pass, the datasheet declares the gap; CPT wired at O1. O3 3b decided 15 Aug 2026: the tilt is ONE dial, P(child \| archetype) = logistic(logit(marginal) + tau*s_a) on the shared centred archetype ladder at `tables.yaml` meta.archetype_ladder, route `inferred`. tau = 0.13, giving iOS 0.677 for whale down to 0.554 for inactive. tau = 0 reproduces a parentless draw exactly, so the sweep 0 / 0.5 / 1 / 2 has a true null |
| 5 | A4 | os_version | observable | user | user_pool | A3 | T2 nested | draw from versions[os] pmf | `device.os_versions` iOS 0.55 / 0.30 / 0.10 / 0.05, Android 0.30 / 0.35 / 0.25 / 0.10, at `market.yaml:145-155`, route `inferred` (O3 3a), source `t9_sim/calibration/provenance_table.md:39-40`, whose own source string reads PLACEHOLDER pending StatCounter | R10 per os | `user_profiles.py:53-57` live | keep; route and value both settled 15 Aug 2026, no longer a hole (§5.3's one remaining hole is `spend_model.scale`); the datasheet declares the placeholder. `tables.yaml` `A4_osversion_given_os` consumes these same numbers, one object, keep the two equal |
| 6 | A5 | device_type | observable | user | user_pool | LU1 | T2 | draw from the archetype's device row; raked so the 0.90/0.10 marginal survives (O1, wired 15 Aug 2026) | `device.device_split` 0.90 / 0.10, route `inferred` (O3 3a, decided 15 Aug 2026: never externally checked, and will not be) | IPF rake preserving the 0.90 phone / 0.10 tablet device marginal to 1e-6 (`tables.yaml` `A5_device_given_archetype`), so the LU1 tilt moves the device split only within each archetype and leaves the population device mix unmoved | `user_profiles.py:49-51` live; its CPT is dead | keep; no external sourcing pass, the datasheet declares the gap; CPT wired at O1. O3 3b decided 15 Aug 2026: the tilt is ONE dial, P(child \| archetype) = logistic(logit(marginal) + tau*s_a) on the shared centred archetype ladder at `tables.yaml` meta.archetype_ladder, route `inferred`. tau = 0.13, giving tablet 0.139 for whale down to 0.088 for inactive. tau = 0 reproduces a parentless draw exactly, so the sweep 0 / 0.5 / 1 / 2 has a true null |
| 7 | LU2 | lu2_click_prop | latent | user | user_pool | LU1 | T3 | Beta(a,b) from a parameter table keyed by archetype | `market.yaml archetype_propensity.click` `archetypes.yaml:27-47`, inferred ordering | R9 pins the population CTR | `user_profiles.py:74` live | keep; T3 not T2, the payload is parameters not a pmf |
| 8 | LU3 | lu3_install_prop | latent | user | user_pool | LU1 | T3 | Beta(a,b) per archetype | `market.yaml archetype_propensity.install` | R9 (click to install) | `user_profiles.py:75` live | keep |
| 9 | LU4 | lu4_payer_prob | latent | user | user_pool | LU1 | T3 | Beta(a,b) per archetype, fixed at zero for inactive (tied to LU5's fixed zero) | `market.yaml archetype_propensity.payer` | R9; exact zero identity on inactive | `user_profiles.py:76,80-83` live | keep; the forced value is one cell of one law, not a second mechanism |
| 10 | LU5 | lu5_ltv_mult | latent | user | user_pool | LU1 | T3 | LogNormal(mu, sigma) per archetype, fixed at zero for inactive. Plain: spend multiplier is lognormal per archetype and fixed at zero for inactive users. Casual players are the reference at multiplier one, so the six dollar median anchor lives only in the base LTV curve and is never counted twice. | `market.yaml archetype_ltv_multiplier` `archetypes.yaml:56-71`; mu solved vs whale-share target | R9 (whale share 0.55–0.65, payer median) | `user_profiles.py:77-85` live | keep |
| 11 | LU6 | lu6_casual … lu6_hypercasual (4 cols) | latent | user | user_pool | LU1 | T3 | one Dirichlet(20 × centroid[archetype]) draw stored as four columns | `market.yaml archetype_interest` `archetypes.yaml:79-86`, inferred | centroids sum to 1 | `user_profiles.py:86-87,92-93` live | keep; one node, not four |
| 12 | user_id | user_id | auxiliary | user | user_pool | none | T4 | pool index, row lineage | design | none | `user_profiles.py:32` live | keep; lineage only, not ground truth |

### 3.2 pool-apps — the app catalogue is built once

| # | Node | Column | Role | Plate | Stream | Parents | Type | Law (one line) | Source | Constraint | v1 status | v2 decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 13 | B1 | app_id | observable | app | app_pool | none | T4 | label `app_{a:05d}`, formed from the app's own index in the catalogue, which is the plate index and so not a parent. An auction inherits it with the rest of the app row, a lookup rather than a parent edge | `profiles.yaml` `default.pool_sizes.<scale>.n_apps`, route `specified`, 150 / 500 / 1200 apps at 100K / 1M / 10M; v1 `profiles.yaml:12-40` | none | `catalogue.py:28-48`; sample `auctions.py:279-293` | keep; the draw is registered in the helper, which closes spec ambiguity 2 |
| 14 | B2 | app_category | observable | app | app_pool | B1 | T4 | quota `n_cat = max(1, round(share × n))` then trim to n; inherited through the app join | `app_categories.share` at `market.yaml:92-97`, Sensor Tower / Newzoo; v1 held the same four numbers under the per-category nesting `app_categories.*.share` `benchmarks.yaml:92-108` | exact quota, no draw | `catalogue.py:30-39` live | keep; law corrected to include the max(1, ·) guard |
| 15 | LA1 | la1_app_quality | latent | app | app_pool | none | T1 | LogNormal(0, 0.30) per app | `entity_latents.sigma_app` `benchmarks.yaml:280`, inferred | unit median | `catalogue.py:44` live | keep |
| 16 | LA2 | la2_whale … la2_inactive (5 cols) | latent | app | app_pool | B2 | T3 | one Dirichlet(3.0 × centroid[app_category]) draw per app, stored as five columns | `market.yaml app_audience.concentration_k` + `market.yaml app_audience.centroids` `market.yaml:274-281`, inferred; carried from v1 `bn.k_audience` and `bn.audience_centroids` `benchmarks.yaml:353-362` | centroid renorm; the global archetype mix is restored by rake R2 | `catalogue.py:54-67`, edge #1 ON | keep; the live concentration is `market.yaml app_audience.concentration_k` = 3.0, and the dead duplicate was v1's `entity_latents.audience_dirichlet_k` = 10.0 (`benchmarks.yaml:283`), inert and deliberately not carried, so there is nothing for S1 to delete (O3) |

### 3.3 pool-campaigns — advertisers, campaigns and their latents are built once

| # | Node | Column | Role | Plate | Stream | Parents | Type | Law (one line) | Source | Constraint | v1 status | v2 decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17 | adv_tier | — | auxiliary | advertiser | campaign_pool | none | T1 | one advertiser forced per tier, the remainder drawn by advertiser share, then shuffled | `market.yaml` `advertiser_scale.advertiser_share` indie 0.85 / mid 0.13 / major 0.02, route `quoted`, count-mix proxy from SocialPeta advertiser counts against Sensor Tower revenue tiers; v1 value at `benchmarks.yaml:145,149,153` | at least one advertiser per tier (constraint block) | `catalogue.py:87-97` live | new node; registering the draw closes spec ambiguity 1 (C1 had no stated law) |
| 18 | C1 | advertiser_id | observable | advertiser | campaign_pool | adv_tier | T4 | label `adv_{idx:03d}` over the shuffled advertiser index | `profiles.yaml` `default.pool_sizes.<scale>.n_advertisers`, route `specified`, 10 / 25 / 50 advertisers at 100K / 1M / 10M; v1 `profiles.yaml:12-40` | none | `catalogue.py:92-97,135` live | keep |
| 19 | C2 | advertiser_scale | observable | advertiser | campaign_pool | C1 | T4 | tier label lookup; its downstream role is the k_cpa key inside H2 | `market.yaml advertiser_scale.k_cpa` indie 0.8 / mid 1.0 / major 1.2, route `inferred`; v1 value at `benchmarks.yaml:147,151,155` | none | `catalogue.py:92-97,136`; k_cpa read `auctions.py:71-72` | keep; the C2 entry point into H2 is now stated (spec ambiguity 5) |
| 20 | C3 | campaign_id | observable | campaign | campaign_pool | C1 | T4 | label `camp_{n:04d}`; counts per advertiser proportional to tier campaign share over tier size, every advertiser guaranteed at least one campaign, rounding drift absorbed into the advertiser with the largest weight | `market.yaml` `advertiser_scale.campaign_share` indie 0.05 / mid 0.20 / major 0.75, route `inferred`, Sensor Tower 2025 IAP spend proxy; pool size from `profiles.yaml` `default.pool_sizes.<scale>.n_campaigns`; v1 `benchmarks.yaml:146,150,154` | guards `catalogue.py:114-117`, per-advertiser minimum `:115` | `catalogue.py:99-142` live | keep; per-advertiser guarantee added to the law |
| 21 | C4 | ad_genre | observable | campaign | campaign_pool | none | T1 | one draw per campaign from `ad_genre_mix` at pool build, frozen thereafter; an auction inherits it with the campaign row, which is a lookup and not a parent edge | `market.yaml` `ad_genre_mix` `market.yaml:132-155`, all 4 leaves route `inferred`. AppsFlyer 2026 anchors casual at about 50 percent of $25B global UA spend; SocialPeta FY2025 creative shares bound strategy and rpg; hypercasual is set deliberately below its supply share. Every figure re-checked 16 Aug 2026 against `docs/Industry_Source_Reports.md` section 21, the June 2026 adversarial verification pass on these exact values. They stay `inferred` because the sources give CREATIVE share and the quantity wanted is SPEND share; v1 `benchmarks.yaml:114-130` | row-level genre exposure preserved by rake R3 at c_idx | `catalogue.py:120-129,137` live | keep; counted once, primary type T1 (entity draw) |
| 22 | LC1 | lc1_creative_appeal | latent | campaign | campaign_pool | none | T1 | LogNormal(0, 0.25) per campaign | `market.yaml entity_latents_extra.sigma_cre` `benchmarks.yaml:281`, inferred | unit median | `catalogue.py:130` live | keep |
| 23 | LC2 | lc2_game_quality | latent | campaign | campaign_pool | none | T1 | LogNormal(0, 0.35) per campaign | `market.yaml entity_latents_extra.sigma_game` `benchmarks.yaml:282`, inferred | unit median | `catalogue.py:131` live | keep; its bid-time role is a build input to c_idx (spec ambiguity 9 settled there) |
| 24 | sample_weight | — (dropped at join) | auxiliary | campaign | campaign_pool | C2 | T4 | campaign share of its tier divided by campaigns in the tier, normalised to sum 1 | derived from `market.yaml` `advertiser_scale.campaign_share` indie 0.05 / mid 0.20 / major 0.75, route `inferred`, spend-share proxy from Sensor Tower 2025 IAP concentration. NB this is campaign VOLUME, not the pool-composition `advertiser_share` that adv_tier draws from; v1 value at `benchmarks.yaml:146,150,154` | sums to 1 (`catalogue.py:150`) | `catalogue.py:144-150`; dropped `auctions.py:308` | keep internal; the functional form is now stated (spec ambiguity 6) |

### 3.4 pool-rivals — eight rivals get six fixed trait tables; no auction ever influences them

All six rows are T1 by the fixed-index rule: rival k, segment s, exchange e and day d are table dimensions, not parents. The per-auction dependence on exchange, device segment and day enters the graph exactly once, in the parent sets of participate_k and b_k, which is where the code reads the tables (`rival_pool.py:112,116`). All six rows are always drawn, dependency #6 being a fixed property of the design (O11), on the dedicated `rival_pool` rng stream.

| # | Node | Column | Role | Plate | Stream | Parents | Type | Law (one line) | Source | Constraint | v1 status | v2 decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 25 | LR1 w_k | — | latent | rival | rival_pool | none | T1 | U(0.35, 0.75) for gaming rivals k < 3, U(0, 0.20) otherwise | ranges `market.yaml rival_pool.value_loading_gaming` / `market.yaml rival_pool.value_loading_nongaming`, hardcoded in v1 at `rival_pool.py:56-57`; K, n_gaming `market.yaml rival_pool.K` / `market.yaml rival_pool.n_gaming`, v1 `benchmarks.yaml:327-332` | R8 declared non-rake; k = 0 is both always-on and gaming, so w_0 > 0 — the premise P4's FKG argument rests on | `rival_pool.py:55-57` | keep the law; the ranges left v1's Python at stage 1 and now sit in `market.yaml` `rival_pool.value_loading_gaming` / `rival_pool.value_loading_nongaming` with value, route and source (O3); v1's rival-pool flag went with the switched-off variants (O11) |
| 26 | LR2 R[s,k] | — | latent | rival | rival_pool | none | T1 | iid N(0,1) array over (segment, rival), the segment index running over the full 2 x 2 product of {iOS, Android} and {phone, tablet} fixed by the domains of A3 and A5, so the array is 4 x 8 under every seed and scale; enters bids scaled by beta_R 0.5 | design; beta_R `market.yaml rival_pool.beta_R` `benchmarks.yaml:327-332`; segment map `rival_pool.py:48-52` | none; the segment dependence enters through b_k only | `rival_pool.py:59-60` | keep; retyped T1 by the fixed-index rule; the segment index is the full 2 x 2 os by device grid fixed by the A3 and A5 domains, so R is 4 x 8 under every seed and scale, not v1's observed-combos map |
| 27 | LR3 pace_k(d) | — | latent | rival | rival_pool | none | T1 | AR(1) day series, p_0 = 0.30·N(0,1), p_d = 0.85·p_{d−1} + 0.30·N(0,1), per rival | `pacing_ar`, `pacing_sigma` `benchmarks.yaml:327-332` | none; mean 0 by construction, and not stationary, p_0 drawn at sigma_p 0.30 against the stationary 0.57, so the variance ramps over the first days of the window | `rival_pool.py:69-76` | keep; fully in config already |
| 28 | LR4 pi_ke | — | latent | rival | rival_pool | none | T1 | U(0.15, 0.60) per (rival, exchange); row k = 0 forced to 1.0 | range hardcoded `rival_pool.py:66` | the forced row guarantees LU7 > 0 and H9 ≥ 1 on every auction | `rival_pool.py:65-67` | keep; the range lives at `market.yaml rival_pool.exch_participation`, moved there at stage 1; retyped T1, the forced row is a fixed cell |
| 29 | LR5 sigma_k | — | latent | rival | rival_pool | none | T1 | U(0.30, 0.55) per rival | range hardcoded `rival_pool.py:63` | none | `rival_pool.py:62-63` | keep; the range moved out of v1's Python at S1 and is now `t9v2/config/market.yaml` `rival_pool.log_bid_dispersion`, route `specified` (O3, closed 15 Aug 2026) |
| 30 | LR6 F_k(d) | — | latent | rival | rival_pool | none | T1 | k = 0 all ones; each k ≥ 1 goes dark with probability 0.5 for one contiguous block, start U{0..n_days−6}, length U{2..5} days | law hardcoded `rival_pool.py:80-84`; length draw `rng.integers(2,6)` `:83` | none | `rival_pool.py:79-85` | keep; move the constants to config (S1); retyped T1; length {2..5} confirmed against code |

### 3.5 join — each auction samples one user, one app, one campaign from the pools

| # | Node | Column | Role | Plate | Stream | Parents | Type | Law (one line) | Source | Constraint | v1 status | v2 decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | pair_idx | — | auxiliary | auction | join | none per row (joint table) | T2 raked joint | one draw of the (archetype, app) cell from the IPF joint; build inputs LA2 and pairing strength 1.0, seed tilt 1 + s·(la2/pi − 1), the seed cells floored at 1e-12 before the rake. Plain: which archetype meets which app is drawn from one joint table, fitted so the overall archetype mix and the equal popularity of apps are both preserved. The app audience profile tilts who meets whom, never how often anyone appears. | dial `market.yaml dependency_strengths.pairing_strength` = 1.0 `benchmarks.yaml:354` | R2: 500-iteration IPF, rows to the archetype mix, columns to uniform 1/A; the mix is a rake target, not a parent (cycle fix) | `auctions.py:144-165`, draw `:279-282`, edge #1 ON | new first-class node; the OFF variant does not exist (app_i goes uniform instead) |
| 32 | app_i | — | auxiliary | auction | join | pair_idx | T4 | app index = pair_idx mod A | design | none | `auctions.py:282` | new node |
| 33 | u_rows | — | auxiliary | auction | join | pair_idx | T3 | uniform member draw inside the drawn archetype's user group | design | none | `auctions.py:283-288` | new node; stays T3 by the stated-table amendment (an implicit uniform over group members is not a CPT) |
| 34 | user_vbin | — | auxiliary | auction | join | LU4, LU5 (via u_rows) | T4 | decile bins of the standardised log of LU4 × LU5, the z-score taken over the positive-value users only; the ~30 percent zero-value users, for whom the log is undefined, are placed below the standardised range, and the tied bin edges collapse so they land together in the bottom bin. Plain: users are sorted into ten equal bins by their latent value, which is payer probability times spend multiplier. Zero value users sit in the bottom bin. | design | none | `auctions.py:174-185`, edge #2 machinery | new node |
| 35 | c_idx | — | auxiliary | auction | join | user_vbin | T2 raked CPT | draw from the per-bin campaign pmf, sample_weight × exp(0.5 · the bin's mean standardised value score · standardised log LC2), row-normalised then IPF raked; build input LC2. Plain: the campaign shown is drawn from a probability table per user value decile, tilted toward better games, and fitted so every campaign still gets exactly its budget share of traffic. | dial `market.yaml dependency_strengths.exposure_beta` 0.5 | R3: 300-iteration IPF, the bin-weighted campaign marginal returns to sample_weight, so budget mix and genre mix survive | `auctions.py:167-197`, draw `:295-303`, edge #2 ON | new node; LC2's role in campaign selection is stated here, which spec 1.6's C-family note designates as the place of record, and it enters as a build-time table input, a parameter edge and not a sampling parent (spec 2.0 scope rule) |

### 3.6 context — exchange, slot, and the clock

| # | Node | Column | Role | Plate | Stream | Parents | Type | Law (one line) | Source | Constraint | v1 status | v2 decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 36 | B3 | ad_exchange | observable | auction | context | none | T1 | per-auction draw from exchange shares | `market.yaml` `context.ad_exchanges`, 2025 US mediation shares, MAX 0.52 route `quoted` and the two tail splits route `inferred`; v1 `benchmarks.yaml:160-165` | R10 | `auctions.py:316-317` live | keep; explicit step 2 assignment closes spec ambiguity 8 for B3 |
| 37 | B6 | slot_format | observable | auction | context | none | T1 | draw from format shares, integer codes banner 1, interstitial 2, rewarded 3 | `market.yaml` `context.slot_format_shares`, route `inferred`, triangulated as revenue share over eCPM (Braberg 2025 against Tenjin/CAS US eCPMs); no measured impression mix is published; v1 `benchmarks.yaml:175-178` | R10 | `auctions.py:318-319` live | keep |
| 38 | _size | — | auxiliary | auction | context | B6 | T2 nested | draw of the WxH label from sizes[format] | `tables.yaml` `size_given_format`, route `inferred`, from `slot_sizes` `benchmarks.yaml:180-189`, Adpiler 2023 / Google Ad Manager / Unity LevelPlay | R10 per format | `auctions.py:321-325` live | new node; B4 and B5 become clean splits |
| 39 | B4 | slot_width | observable | auction | context | _size | T4 | first component of the label split | via _size | none | `auctions.py:326-327` live | keep |
| 40 | B5 | slot_height | observable | auction | context | _size | T4 | second component of the same split | via _size | none | `auctions.py:326-327` live | keep |
| 41 | D4 week | — (no column exists) | auxiliary | auction | context | none | T1 | uniform integer over the four window weeks | `time.window_days` 28 `benchmarks.yaml:288-291` | uniform | `auctions.py:74-76,344` live internal | keep internal, `emitted: false`; was missing from the inventory, added here |
| 42 | D2 | hour_of_day | observable | auction | context | LU1 | T2 raked CPT | draw the hour from the archetype's CPT row. Plain: hour of day is drawn from a 24 hour probability table that depends on the user archetype, fitted so the overall hourly traffic pattern still matches iPinYou. | `tables.yaml` `D2_hour_given_archetype`, base hour marginal from `calibration/ipinyou_hourdow_distribution.csv`, von Mises tilts at `archetype_tilts.mu_kappa` route `inferred` (v1 pairs `bn_cpts.yaml:30-45`, resolved CPT `:151`, build `make_cpts.py:124-125`) | R1: IPF preserves the hour marginal (build assert < 1e-6 `make_cpts.py:282`); table renormalised at load `auctions.py:130-131` | `auctions.py:122-132, 331-339` | keep |
| 43 | D3 | day_of_week | observable | auction | context | LU1 | T2 | draw from the archetype's day-of-week row; raked so the iPinYou dow marginal survives (O1, wired 15 Aug 2026) | `ipinyou_hourdow_distribution.csv` | declared divergence: both marginals survive but the iPinYou hour-by-dow joint is dropped (`auctions.py:123-126`) | `auctions.py:332` live; its LU1 CPT is dead | keep; carry the caveat as a declared divergence so it stops being silent. O3 3b decided 15 Aug 2026: the tilt is ONE dial, P(child \| archetype) = logistic(logit(marginal) + tau*s_a) on the shared centred archetype ladder at `tables.yaml` meta.archetype_ladder, route `inferred`. tau = 0.04, read in value order like A3 and A5, giving weekend 0.310 for whale down to 0.277 for inactive. Corrected 15 Aug 2026: an earlier draft reversed the ladder, but the v7 schema says day-of-week PLAY is near-flat with a modest weekend uptick and that the weekend SPEND skew is carried by t_pay, and v1's own weekend tilt fits tau = +0.038 on this ladder. tau = 0 reproduces a parentless draw exactly, so the sweep 0 / 0.5 / 1 / 2 has a true null |
| 44 | D1 | timestamp | observable | auction | context | D2, D3, D4 | T3 | window start + ((7·week + dow)·24 + hour)·3600 + U{0..3599} | `time.window_start_utc`, design | none | `auctions.py:344-347` live | keep; the second-in-hour jitter makes this T3, not deterministic (verified) |

### 3.7 truth — the oracle computes the true probabilities before anything is priced or drawn

| # | Node | Column | Role | Plate | Stream | Parents | Type | Law (one line) | Source | Constraint | v1 status | v2 decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 45 | r_genre | — | auxiliary | auction | truth | LU6, C4 | T4 | the user's interest weight in the campaign's genre, LU6[ad_genre] | design | none | `auctions.py:358-361` live | added row (the mediator gap fix) |
| 46 | m_stage | — | auxiliary | auction | truth | r_genre | T4 | m_s = (1 − w_s) + w_s·r for stages click, install, pay; w = 0.5, 0.6, 0.7 | `market.yaml funnel_relevance_weights` | none | `auctions.py:362` live | added row |
| 47 | v_slot | — | auxiliary | auction | truth | B6, _size | T4 | format weight times size weight lookup, roughly [0.45, 2] | `market.yaml` `context.slot_quality.format_weight` and `context.slot_quality.size_weight`, route `inferred`, attention proxies with no outside source of any kind; v1 values at `benchmarks.yaml:191-201` | none | `auctions.py:363-364` live | added row |
| 48 | ease | — | auxiliary | auction | truth | B2 | T4 | install ease lookup per app category | `app_categories.*.install_ease` | none | `auctions.py:366` live | added row |
| 49 | mu_cat | — | auxiliary | auction | truth | B2 | T4 | 0.481154 + ln(category LTV tier), minus ln E[plat] | the solved LTV location, written to `calibrated.yaml` by the autocal pass, never hand-set (autocal R9) | R5 carrier: the recentre keeps the LTV mean fixed | `auctions.py:65-66`, recentre `:389` live | added row; R5 now has its carrier node |
| 50 | plat | — | auxiliary | auction | truth | A3 | T4 | 1.8 iOS, 1.0 Android | `market.yaml dependency_strengths.ios_ltv_multiplier` `benchmarks.yaml:365` | R5 | `auctions.py:388-391` live, edge #3 ON | added row |
| 51 | t_pay | — | auxiliary | auction | truth | D2, D3 | T4 | hour-by-dow multiplier table, raked to population mean 1 | `market.yaml dependency_strengths.payer_timing` `benchmarks.yaml:367-371`, inferred | R4 carrier: E[t_pay] = 1 under the joint pmf; approximate where the p_payer clip binds | rake `auctions.py:115-121`, applied `:380-384`, edge #4 ON | added row; R4 now has its carrier node |
| 52 | p_click | p_click | estimand | auction | truth | LU2, v_slot, m_stage, LA1, LC1 | T4 | clip(0.27141 · LU2 · v_slot · m_click · LA1 · LC1, 0, 1) | base_ctr solved (R9) | clip activation measured at the gate | `auctions.py:370-372` live | keep; `ground_truth: true`, hash frozen |
| 53 | p_install | p_install | estimand | auction | truth | LU3, ease, m_stage, LA1 | T4 | clip(2.564963 · LU3 · ease · m_install · LA1, 0, 1); no LC1 term, a deliberate asymmetry vs click | base_ir solved (R9) | clip | `auctions.py:373-375` live | keep; the asymmetry gets an explicit note so S1 does not "fix" it |
| 54 | p_payer | p_payer | estimand | auction | truth | LU4, m_stage, t_pay | T4 | clip(0.257853 · LU4 · m_pay, 0, 1), then times t_pay and re-clipped | base_payer solved (R9) | the clip-vs-rake interaction is a measured quantity | `auctions.py:376-384` live | keep; hash frozen |
| 55 | e_ltv | e_ltv | estimand | auction | truth | mu_cat, LU5, LC2, plat | T4 | exp(mu_cat + sigma²/2) · LU5 · LC2 · plat, sigma 1.648169 | solved (R9, $6 median + whale share) | R5 preserves the mean | `auctions.py:393-394` live | keep; hash frozen |
| 56 | ev_truth | ev_truth | estimand, the oracle | auction | truth | p_click, p_install, p_payer, e_ltv | T4 | the product of the four; inactive users force zero through LU4 = LU5 = 0 upstream | — | none, no tunables | `auctions.py:395` live | keep; this is the retained ground truth the benchmark scores against; hash frozen |
| 57 | z | — | auxiliary | auction | truth | ev_truth | T4 | (log ev − z_mu)/z_sigma on ev > 0 rows, else 0. Plain: the value score is the standardised log of true expected value, using a mean and spread frozen from a warm up sample, and zero on worthless rows. The market reads it. Nothing ever feeds back. | z_mu and z_sigma are solved constants, written to `calibrated.yaml` by the calibration pass, never hand-set. v1 solved both on a 50K warm-up, `t9_sim/config/profiles.yaml:64` | the warm-up statistics are solved constants, not parents | `auctions.py:432-436`, solve `:621-623` live | keep |

### 3.8 market — floor, rivals, our bid, settlement (steps 4 to 9)

| # | Node | Column | Role | Plate | Stream | Parents | Type | Law (one line) | Source | Constraint | v1 status | v2 decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 58 | floor_shape | — | auxiliary | auction | market | none | T1 | resample from the iPinYou floor pmf, normalised by the PAYING median and NOT by its own, so this shape's median is 40/70 = 0.571; with the atom at 0 (about 14 percent of floors are zero) | `calibration/ipinyou_floor_bid_distribution.csv` + `ipinyou_price_summary.csv` median (`config_loader.py:132-135`) | R10 | built `auctions.py:56-59`, drawn `:425-426` live | new row; closes spec ambiguity 3, the spec named this variable and never defined it |
| 59 | H1 | floor_price | observable | auction | market | floor_shape, B6 | T4 | floor_shape × ecpm target of the format; zero shapes stay zero | `ecpm_targets_usd`, Tenjin / Appodeal | the inherited atom at 0 | `auctions.py:426` live | keep; the draw now lives in the registered helper |
| 60 | pay_shape | — | auxiliary | auction | market | none | T1 | resample from the iPinYou paying-price pmf, normalised by the paying median, the SAME denominator row 58 uses, so this shape's median is 1 | `calibration/ipinyou_paying_distribution.csv` + the same median | R10 | built `auctions.py:55,58`, drawn `:428-429` live | new row; closes spec ambiguity 4 |
| 61 | base_e | — | auxiliary | auction | market | pay_shape, B6 | T4 | max(pay_shape × ecpm target of the format, 0.01 × target), the shared DSP-predictable price core feeding every rival bid | `calibration/ipinyou_paying_distribution.csv` + the `ipinyou_price_summary.csv` paying median (`config_loader.py:132-135`), scaled by `ecpm_targets_usd` (`market.yaml:63-75`), Tenjin / Appodeal | zero guard | `auctions.py:428-430` live | keep; gets the explicit step `4 floor` (spec ambiguity 8, it previously sat there only by code position) |
| 62 | participate_k | — | auxiliary, plate(k) | auction | market | LR4, LR6, LR3, B3, D1 (day) | T3 | Bernoulli(clip(pi_ke[k, exchange] · F_k(day) · (0.5 + sigmoid(pace_k(day))), 0, 1)); k = 0 forced true. Plain: each rival flips a coin to enter the auction, with odds set by its taste for this exchange, whether its campaign is live that day, and its budget pace. Rival zero always enters, so every auction has at least one competitor. | design (v10) | clip to [0,1]; forced k = 0 cell | `rival_pool.py:110-115`, ON in paper | O3 3e DECIDED 15 Aug 2026: the day index anchors to the start of the 28-day window, never to the per-chunk minimum timestamp. v1's chunk-relative anchor is a defect and v2 does not reproduce it. The generator asserts the anchor and refuses an out-of-window day index. Build item at stage 2. O3 3c settled the naming on 15 Aug 2026: `participate_k` is the one node name, spec §1.6 was renamed from `Z_k`, and the formal symbol `A_ik` in spec §2.5 is bound to it in that section's notation key |
| 63 | b_k | — | auxiliary, plate(k) | auction | market | base_e, z, LR1, LR2, LR3, LR5, A3, A5 | T3 | log b_k = log base_e + rho·(w_k·z + 0.5·R[s(os, device), k] + pace_k(day)) + sigma_k·N(0,1). Plain: each participating rival bids lognormally around the shared price core, pulled by how much it values this user, its retargeting taste for this device segment, and its budget pace that day. The dial rho scales how much of that private signal enters. At zero every rival bids noise around the core. | rho 0.8, externally anchored (`docs/v10_Anchor_Bands.md`); O3 3d decided 15 Aug 2026 that its one home is `t9v2/config/profiles.yaml` `default.rho`, route `auto-calibrated`, rho being located on a grid so the C3 minus C1 price-RMSE gain `price_rmse_all` lands in the Wang et al. 2023 band [0.05, 0.13]. No copy in `market.yaml`, and no per-run override of a 0.0 default (that was v1, `benchmarks.yaml:224`, recorded in the manifest `simulate.py:95-102`) | R8 declared non-rake, backstopped by the k_global pin R7 and the validation gates | `rival_pool.py:116-119`, ON in paper | new row; the A3 and A5 edges are now explicit parents, not a parenthesis — P2 and P4 quantify over the segment coupling and need them |
| 64 | LU7 | lu7_competing_bid | latent, retained ground truth, never a model feature in any condition (spec :61) | auction | market | b_k, participate_k | T4 plate reduction | the maximum of b_k over participating rivals; never empty because k = 0 always participates | — | none by design; the level is reabsorbed by R7 | paper path `rival_pool.py:122` + `auctions.py:457-467`; v7 default path `auctions.py:509-514` | keep T4, the max over participating rivals, as the node's one law; `ground_truth: true`, hash frozen, never-stochastic guard |
| 65 | H9 | bid_density | observable (SSP-visible, C3 and C4 only) | auction | market | participate_k | T4 plate reduction | the participant count, domain {1..8} | — | none | `rival_pool.py:123` + `auctions.py:467`, ON in paper; absent column when OFF | keep; formula pinned (deposited column); delete the v8 Poisson variant (S7); the code comment at `auctions.py:507` mislabels that column H5 (a v1 code comment; unrelated to node H5 below) |
| 66 | eltv_b2 | — | auxiliary, the bid value input | auction | market | B2 | T4 | E[ltv given category] = exp(mu_cat + sigma²/2) per category, the DSP's observable value estimate, deliberately not the oracle e_ltv | `ltv.*` (R9) | none | table `auctions.py:69-70`, used `:517` live | new row with a load-bearing note: the observable-estimate-versus-oracle gap is what the ablation measures |
| 67 | H2 | bid_price | observable | auction | market | C2, B6, _size, B2, v_slot, ease, eltv_b2 | T3 | location = k_global · k_cpa(C2) · v_slot · ease · eltv_b2 · shade, times LogNormal(0, 0.30); no LU7 input, no latent input. Plain: our bid is a fixed markup times the category average LTV and slot and ease weights, scaled by the advertiser tier aggressiveness, times lognormal exploration noise. The bidder never sees the rival bid or the oracle. | `market.yaml` `auction.shade` 0.85 route `specified`, `auction.sigma_explore` 0.30 route `inferred`; k_global bisected to win rate 0.30 (`auctions.py:610-641`) | R7; P1 gate: exploration independent of value given x, delivered by the rng stream split (main stream for H2 noise, spawned stream for the plate, `rival_pool.py:96`) — a machine-checkable gate, not a comment | `auctions.py:516-520` live; explore slice `:521-530` inert (arm abandoned) | keep; C2 enters through k_cpa, now stated (spec ambiguity 5); the abandoned explore_traffic branch is never built in v2 (rebuild plan's never-rebuild table), and it is not the live sigma_explore noise the law keeps |
| 68 | H5 | min_winning_price | observable (SSP-visible, C3 and C4 only) | auction | market | LU7, H1 | T4 | max(LU7, H1); present on every row, no NaN. Won gives max(LU7, H1), sold to a rival gives LU7, unsold gives H1 | — | none; ≥ H1 by construction | absent in v1; reinstates v1's abandoned arm B1 `min_bid_to_win`, in an all-rows form rather than won-rows-only | new row, 22 Aug 2026. TARGET of the Tier-2 price head, never a per-row feature in any condition, since `won = 1[H2 ≥ H5]` is an identity |
| 69 | H3 | won | outcome (Tier-2 label, settlement truth) | auction | market | H2, LU7, H1 | T4 | 1[H2 ≥ H5], i.e. 1[H2 ≥ max(LU7, H1)] | level pinned via `market.yaml win_rate.target` 0.30 (`profiles.yaml:62-64`) | R7 | `auctions.py:542-544` live | keep; `ground_truth: true`, hash frozen |
| 70 | sold_lost | — | auxiliary | auction | market | H3, LU7, H1 | T4 | (1 − won) · 1[LU7 ≥ H1], the branch selector for H4 | — | none | `auctions.py:543` live | new row (the spec carried it only as an internal, :175) |
| 71 | H4 | winning_price | observable (settlement) | auction | market | H3, sold_lost, H2, LU7 | T4 piecewise | won → H2 (first price, own bid); sold to a rival → LU7; unsold → NaN | — | none | `auctions.py:545-548` live | keep; NaN is the legal value of the unsold branch, meaning no transaction occurred, not missingness; ground-truth freeze extended to H4 (the economics score on it) |

### 3.9 funnel — outcomes are drawn on every row, won or lost; no funnel node reads won

| # | Node | Column | Role | Plate | Stream | Parents | Type | Law (one line) | Source | Constraint | v1 status | v2 decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 72 | E1 | click | outcome | auction | funnel | p_click | T3 | Bernoulli(p_click) | level set upstream by R9 | none of its own | `auctions.py:556` live | keep |
| 73 | E2 | click_timestamp | outcome | auction | funnel | E1, D1 | T3 | no click → −1; click → floor(D1 + Exponential(mean 30 s)) | `market.yaml funnel_delays.click_delay_mean_s`, inferred | −1 is a sentinel branch, not missing data | `auctions.py:569-570` live | keep |
| 74 | F1 | install | outcome | auction | funnel | E1, p_install | T3 | no click → 0; click → Bernoulli(p_install) | R9 | gate identity exact: install = 0 wherever click = 0 (acceptance test) | `auctions.py:557` live | keep |
| 75 | F2 | install_timestamp | outcome | auction | funnel | F1, E2 | T3 | no install → −1; install → floor(E2 + LogNormal(7.5, 1.0)) | `market.yaml funnel_delays.install_delay`, inferred (about 30 min median) | −1 is a sentinel branch, not missing data | `auctions.py:571-573` live | keep |
| 76 | G1 | is_payer | outcome | auction | funnel | F1, p_payer | T3 | no install → 0; install → Bernoulli(p_payer) | R9 | gate identity exact | `auctions.py:558` live | keep |
| 77 | G2 | ltv_value | outcome | auction | funnel | G1, mu_cat, LU5, LC2, plat | T3 | non-payer → 0; payer → LogNormal(mu_cat, 1.648169) · LU5 · LC2 · plat; the declared 90-day post-install total | R9 anchors. `market.yaml` `ltv.base_median_usd` $6, which `ltv.lognormal_mu` is solved to, and the whale top-5-percent share band 0.55-0.65, which `ltv.lognormal_sigma` is solved to. BOTH are solver outputs written to `calibrated.yaml` at stage 2, route `auto-calibrated`; sigma was relabelled from `specified` on 16 Aug 2026 on v1's record at `provenance_table.md:12` | R5: the mu_cat recentre holds the LTV mean fixed under plat, dependency #3, which v2 always applies (O11) | `auctions.py:560-563` live | keep; the zero atom is one branch of one law, not a second mechanism |
| 78 | G3 | ltv_7d | outcome | auction | funnel | G2 | T4 | 0.40 × G2 | `market.yaml` `ltv.decay_d7` 0.40, route `specified` (the 7-day recognition point, spec 2.4's G2 horizon row) | formula pinned (deposited column) | `auctions.py:581` live | keep deterministic; gate is exact recomputation |
| 79 | G4 | ltv_30d | outcome | auction | funnel | G2 | T4 | 0.70 × G2 | `market.yaml ltv.decay_d30` 0.70, route `specified` (the 30-day recognition point on the 90-day curve, stated at spec 1.6 G4) | formula pinned | `auctions.py:582` live | keep deterministic |

### 3.10 Dead rows (excluded from the census)

v1's 8 variant-only and inert rows are gone. The switched-off variants have no place in v2 (O11), and the inert v1 code is recorded in the specification's O2 rather than here, since v2 inherits no code.

**Nothing is dead any more.** The 3 CPT candidates that sat here, os, device_type and day_of_week
given archetype, were **wired by open item O1** on 15 August 2026 and are now live T2 nodes carrying
rows 4, 6 and 43 of the register above. Their tilt strengths were set by O3 3b. This table is kept
empty rather than deleted so the trail is visible.

| # | Node | Type | What it is | v1 status | v2 decision |
|---|---|---|---|---|---|
| — | *(none)* | — | the 3 former candidates moved into the live census | `bn_cpts.yaml:47/70/93` | **WIRED at O1, 15 Aug 2026.** No longer excluded, and the T2 count rose from 6 to 9 |

---

## 5.3 The gate 1 hole list

**Rewritten 15 August 2026.** This section used to list 25 unsourced constants. **All 25 were traced
and their provenance adopted on Ken's instruction, so one hole remains.** The evidence trail, one row
per former hole with its value, route, source and an openable citation, is in
**`docs/T9Sim_Hole_Provenance.md`**. Do not re-derive it here.

Regenerate this section's counts with `t9v2/tools/report_holes.py` after any settings change.

### The one remaining hole

| Setting | Node | Why it is still open |
|---|---|---|
| `training.yaml` `spend_model.scale` | Tier-1 head 4, E(spend given payer) | No document supplies it, and no search will. The prior question is whether the parameter exists at all: v1's spend head is a plain regression scored on `rmse`, which has no scale. Decide at stage 4, per specification open item **O12**. v1's `AFT_SCALE = 1.0` is a DIFFERENT setting, the evaluator's win-curve sigma, and must not be copied here by analogy |

### Where provenance now lives

Inside the settings themselves, as `{value, route, source}` on every calibratable leaf.
`t9v2/tools/gate_report.py` renders the readable view to `t9v2/docs/provenance.md`. Current
state after the 16 August review: **154 leaves, being 16 quoted, 7 auto-calibrated, 26 specified,
104 inferred and 1 hole.**

### Nine settings that had no key at all (16 August 2026)

The multi-agent row review found a class the hole list could not have contained, because a hole is a
key whose value is missing and these had **no key**: the four pool sizes `n_users`, `n_apps`,
`n_campaigns` and `n_advertisers`, plus `advertiser_share`, `campaign_share`, `ad_exchanges`,
`slot_format_shares` and `slot_quality`. Rows B1, C1, adv_tier, sample_weight, B3, B6 and v_slot all
cited them, and `pair_idx`'s law decodes its drawn cell by integer division over `n_apps`, so the
generator could not have run. All were recovered from v1 with their provenance and written to
`profiles.yaml` `default.pool_sizes` and to `market.yaml`, and the Source cells above now point at the
v2 keys rather than v1's. Full account at `t9v2/docs/SPEC_GAPS.md` **SG6**.

They hid because the detector kept only backticked tokens containing a dot and every one of these is
cited by its bare name. That filter is now widened, screened against node ids and column names.

Three of those routings are worth knowing about.

- **`ltv.base_median_usd` was relabelled `quoted` to `inferred`.** v1 cited GameAnalytics and Wappier
  for the $6 median payer spend; neither published such a figure. Note the circularity this sits in:
  `lognormal_mu` is solved against the $6 and the validator then scores the simulator against it.
- **`encoders.shrink` is ONE constant serving both encoders**, as it was in v1 where it was `HCP_EB_K`
  at 20.0. An early v2 draft split it per encoder; open item O13 reversed that on 15 August 2026,
  because empirical-Bayes shrinkage already adapts to sample size and a second constant would have put
  a free knob exactly where the C3 minus C1 result lives. The asymmetry between the two encoders is
  what each averages, not how hard each is shrunk.
- **`direction_checks` were transcribed from v1's code, not its prose**, which had drifted from the
  implementation in 2 places and was inert documentation the validator never read.

### The detector's former blind spot, now closed

`report_holes.py` used to count only the literal string `HOLE`, so a value whose key was **never
written** was invisible to it. It now reports 3 classes: `HOLE`, a key whose value no document
supplies · `UNSOURCED`, a value carrying no route or source · `UNDECLARED`, a key this register
names that the config has no home for.

The third class found **22 keys on 15 August 2026, and all 22 are now closed.** They were not
review items: about 15 were hard stage-2 blockers, because `graph.yaml` declared the node while
nothing stated its law's parameters. LU2 knew it was a Beta keyed by archetype and could not be
drawn.

| What they were | How many | What was done |
|---|---|---|
| LU2 to LU6 per-archetype parameters, LC1 and LC2 spreads, LA2's concentration and centroids | 10 | filled from `t9_sim/calibration/provenance_table.md` rows 39-50 and v1's `archetypes.yaml`, each carrying its own route and source |
| the relevance weights, the 2 delay laws, the 28-day window, the win-rate pin | 6 | filled from the design documents, route `specified` where the spec states them |
| the 3 dependency constants under v1's `bn.*` edge-flag names | 3 | the flags are gone under O11 but the constants are real, so they now live at `market.yaml` `dependency_strengths` |
| the 2 price-shape CSVs and the solved LTV location | 3 | not gaps. The CSVs are inherited under `t9v2/calibration/`; the solved location belongs in `calibrated.yaml` after the solve |

**The structural fix that came with it.** This register's Source column used to name only where a
value CAME FROM, which is why a filled key still read as undeclared: nothing said where the value
now LIVES. Every Source cell above was repointed to the v2 key that holds it. A future gap of this
kind now shows up as a broken pointer rather than as silence.

**Current state: 1 hole, 0 unsourced, 0 undeclared.** The hole is `spend_model.scale`, open item
O12, and it is a stage-4 design decision rather than a missing number.

**One thing the original search exposed, still true.** The calibration bands were nearly recorded as
missing because the search covered only the 5 documents the rebuild plan names as build inputs. They
live in 2 documents outside that pack. Either the pack widens to include them or the bands move into
the specification, which is a gate 1 decision.
