# T9Sim Symbol Plan

11 August 2026. The rules for forming a symbol, the full old-to-new mapping, the newly formed symbols, and the files that carry them.

---

## 1. How a symbol is formed

Klimis notation is the only system. Most quantities in the specification have no paper symbol, so most symbols in section 2 are newly formed. 9 rules.

**R1.** Where the paper has a symbol for the quantity, the specification takes it unchanged.

| Item | Paper form | Specification form |
|---|---|---|
| 5 overloaded letters | `q`, `π`, `κ`, `α`, `ρ` | see R7 |
| censoring operator | `𝒞_y^k`, `𝒞_m^k` | `𝒞^k`, one operator |
| per-row record | `d_i^full`, `d_i^{obs,k}` | `𝒟_i^full`, `𝒟_i^{obs,k}`, because `d_i` is the day index |
| rival structure dial | `ρ_r` | `ρ` |
| funnel block | `y_i`, 3 outcomes | `y_i`, widened to the 8 funnel columns |
| supply pool bundle | `z_i^s` | `z_i^s`, kept, though the newly formed app tag is `a` |

**R2. Seven shapes.**

| Shape | Example |
|---|---|
| bare symbol | `θ_i^click` |
| column address, one per observable column | `x_i[app_category]` |
| function of a row-derived key | `β_0(x_i)`, `μ^ltv(x_i[app_category])`, `p_k(d_i)` |
| row-to-key map | `s^u(i)`, `s^size(i)` |
| conditional pmf | `χ^city(· \| x_i[region])` |
| indexed family member | `p_i^{(stage)}` |
| bracket lookup of one component of a simplex or table | `χ_i^interest[x_i[ad_genre]]`, `centroid[κ_u]` |

2 columns take a bare letter instead of a bracket address, `t_i` for D1 and `e_i` for B3. Both are newly formed. Both are members of `x_i` and are always written by their letter, never as a bracket address. `d_i` and `s(i)` are derived keys, not columns. A law that reads a column at pool level, before rows exist, uses the pool-level column address `x_a[app_category]`, parallel to `x_i[…]`.

**R3. Scripts.**

| Position | What it carries |
|---|---|
| superscript, lowercase tag | the name, never a power. A power goes in brackets, so `σ²` becomes `(σ^ltv)²` |
| superscript, word in parentheses | a free index over a named family, `p_i^{(stage)}` |
| subscript, single | the index. `i` row, `k` rival, `j` variable, `u`/`a`/`c` pool entity, `0` baseline or always-on or a fixed realised value |
| subscript, compound | `ik` row then rival, `s(i),k` row-derived key then rival |
| subscript, range | `1:n` all rows, `1:K` all rivals |

A newly formed name always goes in the superscript. 6 inherited symbols keep the paper's subscript placement, `σ_k`, `π_k`, `p_k`, `F_k`, `R`, `β_R`. Two more, `φ_p` and `σ_p`, take that placement by analogy but are newly formed (section 3.2): the paper names the day-level pacing process without giving its persistence or its innovation scale a symbol. A pool latent takes its pool index at pool level and `i` inside a row law, `λ_u^ltv` / `λ_i^ltv`. A sentence stating a row consequence uses the row form `_i` throughout. Case is part of the symbol and no 2 symbols differ by case alone.

**R4. Tags.** A tag is lowercase.

| Tag source | Example |
|---|---|
| parquet column or config key in full, underscore kept | `χ^os_version` |
| inherited paper tag | `q^app` |
| plain word | `explore`, `clearing`, `size`, `delay` |
| one of 10 short codes, `mkt` `riv` `cat` `camp` `cpa` `ecpm` `os` `7d` `30d` `val` | `v_i^mkt` |
| compound, joined with a comma | `𝒟_i^{obs,k}`, `μ^{delay,click}` |
| pool, `u` user, `a` app, `c` campaign, `m` rival market | `z_c^c` |

**R5. Base letter by kind.** Chosen by what kind of quantity it is, never by which pool it came from. Closed list. Where a letter carries 2 jobs, both are named. A letter goes to R7 only when both meanings can appear in the same law.

| Letter | Kind |
|---|---|
| `x` | observable bid-time column |
| `t`, `τ` | event time, reporting delay |
| `d`, `e` | day index (week `d_i^week`), exchange index |
| `s` | row-to-key map with a key tag, and bare `s(i)` the observable segment |
| `c`, `a`, `r`, `y` | click label, install label, revenue, funnel block and its tagged members such as `y_i^pay`. `c` is also the campaign pool index and `a` the app pool index |
| `b`, `m`, `f`, `u`, `w`, `N`, `h` | own bid and the free bid argument of the win curve, winning price, floor, top competing bid, win, rival count, market block |
| `B`, `A`, `𝒜` | rival bid, rival participation indicator, participant set |
| `z` | latent state bundle, latent value index |
| `θ` | latent propensity, probability-valued |
| `q` | latent quality. NB `q_i^slot` here is the observable derived slot weight `v_slot`, not the paper's latent slot quality, which has no counterpart here; `q_i^app` is the paper's latent and maps to LA1 |
| `κ` | archetype label |
| `χ` | distribution over categories |
| `λ` | multiplier on money |
| `α` | multiplier on a response probability, latent appeal `α^creative` |
| `δ` | revenue recognition fraction |
| `ω` | loading or weight |
| `β` | funnel base level when tagged `β_0^click`, price baseline when given an argument `β_0(x_i)`, scale coefficient `β_R`. Bare `β_0` is never written. NB the tagged form is a **multiplicative base rate on the probability scale**, not the paper's log-odds intercept of the same spelling; the argument form `β_0(x_i)` is the paper's and keeps the paper's meaning |
| `μ`, `σ` | location, dispersion |
| `φ`, `ξ` | AR persistence, Dirichlet concentration |
| `ζ`, `ψ` | own-bid scale dial, price shape factor |
| `ρ`, `γ` | market structure dial, value-awareness dial. Bare `ρ` is the paper's rival dial, subscripted r there. The paper's other bare `ρ`, the ROAS threshold, is `T^roas` here |
| `R`, `π` | rival segment shift matrix, rival participation propensity |
| `g` | logistic pacing map |
| `T` | calibration target |
| `n`, `K` | count, rival pool size |
| `p` | true probability, density or conditional law `p(·)`, rival pacing state `p_k(d_i)` |
| `v`, `ℓ` | expected value, expected revenue per payer |
| `F` | distribution function, rival flight mask `F_k(d_i)` |
| `ε`, `η` | noise draw, exploration draw |
| `𝒵`, `𝒟`, `𝒞` | latent space, data table with `𝒟_i` one of its rows, censoring operator |
| `𝒢`, `𝒫` | genre set, generator class |
| `𝕀`, `𝔼`, `Pr`, `Cov`, `Var` | indicator, expectation, probability, covariance, variance |

**R6. Marks.** `⋆` oracle, applied only where the paper applies it, so `v_i^⋆` carries it and the true head probabilities do not. `^` estimated from data. `~` visible under a view, always set over the symbol, so the infix `∼` reads "is sampled from". Nothing else decorates a symbol. `∅` is the unobserved cell, `⊥` a quantity that does not exist on that row, `{}` the empty set.

**R7. Overloaded letters.**

| Letter | Specification meaning | Meaning moved out of the letter | New symbol for the moved meaning |
|---|---|---|---|
| `q` | latent quality `q^app`, and the observable slot weight `q^slot` | win-probability model | `F_{u\|x}(b)`, fitted `F̂_{u\|x}(b)`; the paper's own symbol for that curve is `F̂_m(b \| x_i)` (2.4) |
| `π` | rival exchange participation `π_k(e_i)` | user archetype | `κ_i` |
| `κ` | user archetype `κ_i`, a destination not a split | rival competition intensity | `N_i` |
| `α` | fit and appeal `α^creative`, `α^camp` | own-bid and logging scale | `ζ^shade` |
| `ρ` | rival private-structure dial, anchored `ρ^anchor` | ROAS target | `T^roas`, reserved. ROAS is not in the specification |

The "specification meaning" column says which meaning this document gives the letter. It does not assert that the paper gives it the same meaning. Two rows differ from the paper and are recorded in 2.3: the paper writes `π_i` for the user archetype and `κ_i` for rival competition intensity, and the paper's `N` already names realised competition density, so `N_i` is inherited rather than fresh.

**R8. Names stay names.** Schema codes (`A1`-`H9`, `LU`/`LA`/`LC`/`LR`), parquet columns, config keys and Python identifiers do not change. Inside a formula the code is replaced by its symbol. In the section 2 mapping a `Current` entry in backticks is one of these names. It is not edited anywhere. Only its use inside a formula is replaced. An unbackticked `Current` entry is a math symbol and is replaced everywhere it appears. A constant rename applies in formula positions only. A backticked name in a prose or note cell stays as it is.

**R9. One written form.** Distributions are proper names and transforms are lowercase words, `Bernoulli`, `Categorical`, `Beta`, `LogNormal`, `Normal`, `Uniform`, `Dirichlet`, `Exponential`, `vonMises`, `Poisson`, `clip`, `max`, `min`, `floor`, `exp`, `log`, `logit`, `zscore`, `quantile`, `centroid`, `pa`. Sets and spaces are calligraphic, `𝒜 𝒟 𝒞 𝒢 𝒫 𝒵`. Operators are upright, `𝕀 𝔼 Pr Cov Var`. Multiplication is `·`. A ternary is written as a cases brace. One quantity has exactly one written form in a law, in a table and in prose. An inline distribution draw becomes its draw symbol, its law appended once beside the law that uses it, as in E2, F2 and H2. A draw from a conditional pmf is written `∼ χ^tag(· \| key)`. `Categorical(·)` is reserved for unconditioned share vectors.

---

## 2. The mapping

One row per quantity. `Current` lists every spelling the specification uses for it. A backticked `Current` entry is a name in the code or schema and stays as it is (R8). `Code` is the schema code where the quantity has one.

### 2.1 Observable columns (A, B, C, D)

| Current | Meaning | Code | New |
|---|---|---|---|
| `A1` | region | A1 | `x_i[region]` |
| `A2` | city | A2 | `x_i[city]` |
| `A3` | os | A3 | `x_i[os]` |
| `A4` | os_version | A4 | `x_i[os_version]` |
| `A5` | device_type | A5 | `x_i[device_type]` |
| `B1` | app_id | B1 | `x_i[app_id]` |
| `B2` | app_category | B2 | `x_i[app_category]` |
| app_category at pool level | pool-level column address, read in LA2 before rows exist | B2 | `x_a[app_category]` |
| `B3` | ad_exchange | B3 | `e_i` |
| `B4` | slot_width, a component of `s^size(i)` | B4 | `x_i[slot_width]` |
| `B5` | slot_height, a component of `s^size(i)` | B5 | `x_i[slot_height]` |
| `B6` | slot_format | B6 | `x_i[slot_format]` |
| `C1` | advertiser_id | C1 | `x_i[advertiser_id]` |
| `C2`, `adv_tier` | advertiser_scale | C2 | `x_i[advertiser_scale]` |
| `C3` | campaign_id | C3 | `x_i[campaign_id]` |
| `C4` | ad_genre | C4 | `x_i[ad_genre]` |
| `D1` | timestamp | D1 | `t_i` |
| `D2` | hour_of_day | D2 | `x_i[hour_of_day]` |
| `D3` | day_of_week | D3 | `x_i[day_of_week]` |
| `D4`, week | week, generator-internal, not a parquet column | D4 | `d_i^week` |
| x (Part III) | the always-visible pre-bid columns A, B, `C`, D and the floor | — | `x_i` |

### 2.2 Label and market columns (E, F, G, H)

| Current | Meaning | Code | New |
|---|---|---|---|
| `E1` | click label | E1 | `c_i` |
| `E2`, `click_ts` | click_timestamp | E2 | `t_i^click` |
| `F1` | install label | F1 | `a_i` |
| `F2` | install_timestamp | F2 | `t_i^install` |
| `G1` | is_payer | G1 | `y_i^pay` |
| `G2`, ltv | ltv_value, 90-day post-install total | G2 | `r_i` |
| `G3` | ltv_7d | G3 | `r_i^7d` |
| `G4` | ltv_30d | G4 | `r_i^30d` |
| `H1`, ℓ, ℓ_i | floor_price, a component of `x_i` | H1 | `f_i` |
| `H2`, b^own, b^own_i | bid_price, own bid | H2 | `b_i` |
| `H3`, y^win, the word "won" used as a condition in a cases law | won indicator | H3 | `w_i` |
| `H4`, y^clr | winning_price | H4 | `m_i` |
| max(LU7, H1), the winning threshold | the minimum price a bid must clear. This is the paper's body `m_i`; this document's `m_i` is the paper's appendix `m`, the H4 clearing price | H1, LU7 | `m_i^win` |
| `H9`, N, N_i, H9_i | bid_density, rival count | H9 | `N_i` |
| `H5`-`H8` | declared unused slots | H5-H8 | no quantity, codes unchanged |
| funnel block, `p(funnel_i \| ·)` | the E, F and G part of a row, `(c_i, t_i^click, a_i, t_i^install, y_i^pay, r_i, r_i^7d, r_i^30d)` | E, F, G | `y_i` |
| market block, `p(market_i \| ·)` | the H and LU7 part of a row, `(f_i, N_i, u_i, b_i, w_i, m_i)` | H, LU7 | `h_i` |

### 2.3 Latents (LU, LA, LC, LR)

| Current | Meaning | Code | New |
|---|---|---|---|
| `LU1`, κ, κ_u, archetype in the pairing law | user archetype | LU1 | `κ_u` / `κ_i` |
| `LU2`, q^clk | click propensity | LU2 | `θ_u^click` / `θ_i^click` |
| `LU3`, q^ins | install propensity | LU3 | `θ_u^install` / `θ_i^install` |
| `LU4`, q^pay | payer probability | LU4 | `θ_u^pay` / `θ_i^pay` |
| `LU5`, m, m_u | LTV multiplier | LU5 | `λ_u^ltv` / `λ_i^ltv` |
| `LU6`, ι, ι_u | genre interest simplex | LU6 | `χ_u^interest` / `χ_i^interest` |
| `LU7`, V, V_i, LU7_i | top competing bid | LU7 | `u_i` |
| `LA1`, `la1`, λ_a | app quality | LA1 | `q_a^app` / `q_i^app` |
| `LA2`, α, α_a | app audience profile simplex | LA2 | `χ_a^audience` / `χ_i^audience` |
| `LC1`, `lc1`, η_c | creative appeal | LC1 | `α_c^creative` / `α_i^creative` |
| `LC2`, `lc2`, χ_c | game quality | LC2 | `λ_c^game` / `λ_i^game` |
| `LR1`, `w_k` | rival private value loading | LR1 | `ω_k` |
| w (bare, in a rival law or a covariance) | the generic rival loading. It resolves to `ω_k` in the general law and to `ω_0` where K = 1 forces the sole rival to be the always-on one | LR1 | `ω_k` / `ω_0` |
| w_0 | loading of the always-on rival | LR1 | `ω_0` |
| `LR2`, R, R[s,k] | rival retargeting level-shifter matrix | LR2 | `R_{s(i),k}` |
| `LR3`, p_k(d), `pace` | rival day-level AR(1) pacing state | LR3 | `p_k(d_i)` |
| `LR4`, π_k(e), `pi_ke` | rival per-exchange participation propensity | LR4 | `π_k(e_i)` |
| `LR5`, σ_k, `sigma_k` | rival log-bid dispersion | LR5 | `σ_k` |
| `LR6`, F_k(d), `flight` | rival flight on/off mask | LR6 | `F_k(d_i)` |

*The archetype letter, recorded against the paper.* The paper writes `π_i` for the user archetype and `κ_i` for rival competition intensity. This document writes `κ_u` / `κ_i` for the archetype, so the paper's `π_i` is this document's `κ_i`, and the paper's latent `κ_i` has no counterpart here because no latent intensity scalar is carried. The observable rival count `N_i` is offered in R7 for that role, but it is the paper's own `N` (realised competition density, an SSP-visible column) rather than a fresh symbol, and it is an observable where the paper's `κ_i` is a latent inside `z_i`.

### 2.4 Estimands and head means

| Current | Meaning | Code | New |
|---|---|---|---|
| `p_click`, ν^clk, P(click) | true click probability | E1 | `p_i^click` |
| `p_install`, ν^ins, P(install\|click) | true install probability given click | F1 | `p_i^install` |
| `p_payer`, ν^pay, P(payer\|install) | true payer probability given install | G1 | `p_i^pay` |
| ν (bare) | any one of the 3 probability heads, stage ∈ {click, install, pay} | — | `p_i^{(stage)}` |
| `e_ltv`, ē_i, E[spend\|payer], E[ltv\|payer] | true expected LTV given payer | G2 | `ℓ_i` |
| `ev_truth`, EV, EV_i, ev | true impression expected value | — | `v_i^⋆` |
| EV_market | supply-only EV channel | — | `v_i^mkt` |
| P(win\|bid), F_{V\|x}(b) | Tier-2 estimand, the conditional win curve | H3 | `F_{u\|x}(b)` |
| fitted win curve | Tier-2 model output | — | `F̂_{u\|x}(b)` |
| the paper's price-distribution model | the paper's own symbol for the Tier-2 curve, conditioned on the price rather than on the top rival bid. It equals `F_{u\|x}(b)` for b ≥ f_i and differs below the floor | H4 | `F̂_m(b \| x_i)` |
| `value_estimate`, `eltv_b2` | the generator's logging bid value function | — | `v_i^proxy` |
| ē_B2 | DSP benchmark E[ltv \| app_category]. A generator-side category lookup, NOT the paper's fitted Tier-1 revenue head `ℓ̂(x_i)`, which is a model output and out of the generator's scope | B2 | `ℓ̂^cat` |

### 2.5 Generator-internal derived quantities

| Current | Meaning | Code | New |
|---|---|---|---|
| `user_id`, `u_rows`, u(i) | user pool position, row lineage | — | `s^u(i)` |
| `app_i`, a(i), app in the pairing law | app pool position | — | `s^a(i)` |
| `c_idx`, c(i) | campaign pool position | — | `s^c(i)` |
| `pair_idx` | (archetype x app) cell of the IPF joint | — | `s^pair(i)` |
| `user_vbin` | user value bin | — | `s^bin(i)` |
| `_size` | drawn width and height key per format, parent of B4 and B5 | B4, B5 | `s^size(i)` |
| g (in `user_vbin`) | user value score, z(log(LU4 · LU5)). The symbol replaces the expression in the exposure law, its defining law `z_i^{u,score} = zscore(log(θ_i^pay·λ_i^ltv))` stated once beside that law | — | `z_i^{u,score}` |
| `sample_weight` | campaign draw weight | — | `ω_c^camp` |
| `r_genre`, r, r_i, ι[genre], LU6[C4] | genre relevance, the lookup `χ_i^interest[x_i[ad_genre]]` | LU6, C4 | `α_i^genre` |
| `m_stage`, `m_click`, `m_install`, `m_pay` | per-stage relevance multiplier | — | `α_i^click`, `α_i^install`, `α_i^pay` |
| m_st, m_s | per-stage relevance multiplier in a generic stage law, stage ∈ {click, install, pay} | — | `α_i^{(stage)}` |
| `v_slot`, v_i | slot quality weight. Observable and derived: a deterministic function of B6 and `_size`, computed by the DSP inside its own bid law. The paper's `q_i^slot` is a latent inside `z_i` and has no counterpart here (R5) | B6 | `q_i^slot` |
| `ease` | install-ease per app category | B2 | `α_i^ease` |
| `t_pay` | payer-timing hour x dow multiplier | D2, D3 | `α_i^time` |
| `plat` | iOS spend multiplier | A3 | `λ_i^os` |
| `mu_cat`, μ_cat | LTV lognormal location per category | B2 | `μ^ltv(x_i[app_category])` |
| `base_e`, base_e(i) | baseline rival price level | — | `β_0(x_i)` |
| `z`, z_i | standardised log EV | — | `z_i^val` |
| `Exp(mean_clk)` draw | click reporting delay | E2 | `τ_i^{delay,click}` |
| `LogNormal(μ_ins, σ_ins)` draw | install reporting delay | F2 | `τ_i^{delay,install}` |
| `LogNormal(0, σ_explore)` factor | exploration draw on our bid | H2 | `η_i` |
| `Z_k`, `Z_ik`, `participate_k` | rival participation indicator | LR4 | `A_ik` |
| `Z_0` | participation of the always-on rival, ≡ 1 | — | `A_i0` |
| `sold_lost`, the words "sold-lost" used as a condition in a cases law | lost by us and sold to a rival | H3 | `w_i^riv` |
| b_ik, b_k | rival k's bid on row i | LU7 | `B_ik` |
| ε_ik, ε (bare) | idiosyncratic bid noise | LR5 | `ε_ik` |
| A_i | participant set, `\|𝒜_i\| = N_i` | H9 | `𝒜_i` |
| max_{k ∈ A_i} | max over participants | LU7 | `max_{k ∈ 𝒜_i}` |
| gate(p) | logistic pacing multiplier | LR3 | `g(·)` |
| `zscore(·)`, z(·) | standardising function | — | `zscore(·)` |
| `qcut` | quantile binning | — | `quantile(·)` |
| `centroid[·]` | Dirichlet centroid lookup | LA2 | `centroid[·]` |
| σ² | LTV lognormal variance term in e_ltv | G2 | `(σ^ltv)²` |

### 2.6 Named constants and dials

| Current | Meaning | Code | New |
|---|---|---|---|
| `base_ctr` | click base rate, 0.27141 | E1 | `β_0^click` |
| `base_ir` | install base rate, 2.564963 | F1 | `β_0^install` |
| `base_payer` | payer base rate, 0.257853 | G1 | `β_0^pay` |
| sigma, σ | LTV lognormal scale, 1.648169 | G2 | `σ^ltv` |
| 0.481154 | mu_cat intercept | G2 | `μ_0^ltv` |
| `w_s` | stage weights, 0.5 / 0.6 / 0.7 | — | `ω^click`, `ω^install`, `ω^pay` |
| w_st | stage weight in a generic stage law, stage ∈ {click, install, pay} | — | `ω^{(stage)}` |
| `decay_d7` | 0.40 revenue recognition | G3 | `δ^7d` |
| `decay_d30` | 0.70 revenue recognition | G4 | `δ^30d` |
| `decay` (the generic constant of the combined G3/G4 law) | one constant standing for both horizons. It has no single new symbol, so the combined law is written as two laws, one per horizon | G3, G4 | `δ^7d`, `δ^30d` |
| `shade` | our-bid shading dial | H2 | `ζ^shade` |
| `k_global` | global bid scaling constant | H2 | `ζ^global` |
| `k_cpa` | CPA bid scaling constant | H2 | `ζ^cpa` |
| σ_explore, `sigma_explore` | exploration spread, scale of `η_i` | H2 | `σ^explore` |
| `floor_shape` | floor price shape | H1 | `ψ^floor` |
| `pay_shape` | winning price shape | H4 | `ψ^clearing` |
| `ecpm_target`, target | per-format eCPM target | B6 | `T^ecpm(x_i[slot_format])` |
| `mean_clk` | click-delay exponential mean | E2 | `μ^{delay,click}` |
| μ_ins | install-delay lognormal location | F2 | `μ^{delay,install}` |
| σ_ins | install-delay lognormal scale | F2 | `σ^{delay,install}` |
| σ_app | LA1 lognormal scale | LA1 | `σ^app` |
| σ_cre | LC1 lognormal scale | LC1 | `σ^creative` |
| σ_game | LC2 lognormal scale | LC2 | `σ^game` |
| `k_aud` | LA2 Dirichlet concentration, 3.0 | LA2 | `ξ^audience` |
| 20 | LU6 Dirichlet concentration | LU6 | `ξ^interest` |
| `n_cat` | exact app count per category | B2 | `n^cat` |
| `n_gaming` | gaming rival count, 3 | LR1 | `n^gaming` |
| `n_days` | flight window length | LR6 | `n^days` |
| `window_start` | 28-day window origin | D1 | `t_0` |
| ρ | private-structure amplitude dial | LR1 | `ρ` |
| ρ* | anchored operating point, 0.8 | LR1 | `ρ^anchor` |
| K, K_riv | number of rival archetypes, 8 | LR1 | `K` |
| β_R, `beta_R` | retargeting shifter scale, 0.5 | LR2 | `β_R` |
| φ_p, `phi_p` | pacing AR(1) persistence, 0.85 | LR3 | `φ_p` |
| σ_p, `sigma_p` | pacing innovation scale, 0.30 | LR3 | `σ_p` |
| eta (LR3 innovation) | pacing innovation, iid Normal(0, 1) | LR3 | `ε_k(d_i)` |
| γ | OFF-path value loading dial | — | `γ^mkt` |
| g (P4 dial) | rival-market value-awareness dial | — | `γ^riv` |
| z_μ, `z_mu` | EV log-mean used to standardise | — | `μ^val` |
| z_σ, `z_sigma` | EV log-sd used to standardise | — | `σ^val` |
| N (pool) | user pool size | LU | `n^user` |
| M | number of auction rows | — | `n` |
| n = 10 (in the empirical notes) | a seed count, not a row count. Because `n` now names the row count, the seed count is written out in words | — | "10 seeds" |

*Two paper dials with no address here.* The paper's feature-informativeness dial, its σ with the x tag at `main.tex:391`, has no counterpart in this specification, which carries no such dial; nothing is minted for it. The paper's own-bid scale, written α in its logging rule at `main.tex:629`, is occupied here by `ζ^shade` together with `ζ^global` and `ζ^cpa`. The paper's campaign fit term, its α with the camp tag, is carried here by `λ_c^game` and `χ^expose` rather than by one symbol.

### 2.7 Categorical share vectors named in the laws

| Current | Meaning | Code | New |
|---|---|---|---|
| π in Cat(π) | archetype shares | LU1 | `χ^archetype` |
| region shares | A1 pmf | A1 | `χ^region` |
| city shares \| region | A2 pmf | A2 | `χ^city(· \| x_i[region])` |
| os shares | A3 pmf | A3 | `χ^os` |
| version shares \| os | A4 pmf | A4 | `χ^os_version(· \| x_i[os])` |
| device shares | A5 pmf | A5 | `χ^device_type` |
| exchange shares | B3 pmf | B3 | `χ^ad_exchange` |
| size \| slot_format | B4, B5 pmf, drawn as `s^size(i)` | B4, B5 | `χ^size(· \| x_i[slot_format])` |
| format shares | B6 pmf | B6 | `χ^slot_format` |
| `advertiser_share` | C2 pmf | C2 | `χ^advertiser_scale` |
| `genre_mix` | C4 pmf | C4 | `χ^ad_genre` |
| P(hour \| κ), von Mises CPT | D2 conditional pmf | D2 | `χ^hour_of_day(· \| κ_i)` |
| dow shares | D3 pmf | D3 | `χ^day_of_week` |
| week pmf | D4 pmf | D4 | `χ^week` |
| `pmf[user value-decile]` | campaign exposure pmf | C3 | `χ^expose(· \| s^bin(i))` |

### 2.8 Joint factorisation (section 2.2)

| Current | Meaning | Code | New |
|---|---|---|---|
| $p(\text{master})$ | joint law of one whole dataset | — | `p(𝒟^full)` |
| $p(\text{pools})$ | joint law of the four pools | — | `p(z^pool)` |
| pools | the four pool bundles together | — | `z^pool` |
| $\theta^U_u$ | latent bundle of user u | LU | `z_u^u` |
| $\theta^A_a$ | latent bundle of app a | LA | `z_a^s` |
| $\theta^C_c$ | latent bundle of campaign c | LC | `z_c^c` |
| $\theta^K_k$ | latent bundle of rival k | LR | `z_k^m` |
| $\theta^K_{1:K}$ | all K rival bundles | LR | `z_{1:K}^m` |
| $\theta_i$ | latent bundle row i drew. Narrower than the paper's `z_i`, which has a fourth component, the rival pool: here the rival bundle is carried separately as `z_{1:K}^m` because it is not row indexed | — | `z_i` |
| $x_{ij}$ | variable j of row i | — | `x_ij` |
| $\mathrm{pa}(x_{ij})$ | parent set of a variable | — | `pa(x_ij)` |
| $\prod_u, \prod_a, \prod_c$ | products over the user, app and campaign pools | — | `∏_u, ∏_a, ∏_c` |
| $\prod_k, \prod_i, \prod_j$ | products over rivals, rows, variables | — | `∏_k, ∏_i, ∏_j` |
| $\mid$ | conditioning bar | — | `\|` |
| $\dots$ | elision | — | `…` |
| $\underbrace$ | factor annotation | — | unchanged |
| $\Big[\ \Big]$, $\left \right$ | brackets and sized delimiters | — | unchanged |

### 2.9 Part III identification symbols

| Current | Meaning | Code | New |
|---|---|---|---|
| O_c | censoring mask of condition c, k ∈ {C1,…,C4} | — | `𝒞^k` |
| **G** | class of generators identification is defined over | — | `𝒫` |
| b | free bid argument of the win curve | H2 | `b` |
| Y | a single funnel outcome, stage ∈ {click, install, pay, ltv} | E, F, G | `y_i^{(stage)}` |
| E[Y \| x] | conditional funnel mean, the MMP target | — | `𝔼[y_i^{(stage)} \| x_i]` |
| n | a realised rival-count value | H9 | `n_0` |
| x_{1:M} | context block over all rows | — | `x_{1:n}` |
| z_{1:M} | latent-value block over all rows | — | `z_{1:n}^val` |
| Cov(log V, z \| x) | value-price covariance | LU7 | `Cov(log u_i, z_i^val \| x_i)` |
| Var(z \| x) | conditional variance of z | — | `Var(z_i^val \| x_i)` |
| K = 1 | single-rival special case | LR1 | `K = 1` |
| N ≡ 1 | collapsed pseudo-rival construction | H9 | `N_i ≡ 1` |
| IPV | independent private values | — | IPV, term unchanged |
| root-n | regular estimator rate | — | root-n, term unchanged |
| P1, P2, P3, P4 | proposition labels | — | `P1`-`P4`, unchanged |

### 2.10 Distributions

| Current | Meaning | Code | New |
|---|---|---|---|
| `Cat(·)` | categorical draw from an unconditioned share vector | — | `Categorical(·)` |
| `Beta(a,b \| κ)` | archetype-conditional Beta | LU2-LU4 | `Beta(a, b \| κ_u)`, row-law variant `Beta(a, b \| κ_i)`, the dot form only where the original writes a dot |
| `LogNormal(μ, σ)` | lognormal draw | — | `LogNormal(·, ·)` |
| LogNormal(μ, σ \| κ) | archetype-conditional lognormal | LU5 | `LogNormal(· \| κ_u)`, row-law variant `LogNormal(· \| κ_i)` |
| `Dirichlet(·)` | simplex draw | LU6, LA2 | `Dirichlet(·)` |
| `Bern(·)`, `Bernoulli(·)` | Bernoulli draw | — | `Bernoulli(·)` |
| `N(0,1)` | standard normal | — | `Normal(0, 1)` |
| `U(a, b)` | continuous uniform | — | `Uniform(·, ·)` |
| `U{a..b}` | discrete uniform | — | `Uniform{· .. ·}` |
| `Exp(mean_clk)` | exponential click delay | E2 | `Exponential(μ^{delay,click})` |
| von Mises CPT | hour-of-day family, law is `χ^hour_of_day` | D2 | `vonMises` |
| AR(1) | pacing process | LR3 | `AR(1)` |
| IPF | iterative proportional fitting for the pairing joint | — | IPF, term unchanged |
| Poisson | exogenous rival count in the contrasting market | H9 | `Poisson` |

One line accompanies each Beta shape law. `a, b` are the per-archetype shape pair from `archetypes.yaml`, parameter slots, not the install label.

### 2.11 Operators, relations, indices, sentinels

| Current | Meaning | Code | New |
|---|---|---|---|
| `clip(·, 0, 1)` | probability clip | — | `clip(·, 0, 1)` |
| `max`, `min` | maximum, minimum | — | `max`, `min` |
| `exp(·)` | exponential | — | `exp(·)` |
| `log`, `ln(·)` | natural log | — | `log` |
| `floor(·)` | integer floor on timestamps | — | `floor(·)` |
| `E[·]` | expectation | — | `𝔼[·]` |
| `P(·)` | probability of an event | — | `Pr(·)` |
| `1[·]`, `1{·}` | indicator | — | `𝕀[·]` |
| `? :` | conditional value in the E2, F2 and G2 laws | E2, F2, G2 | cases brace, `t_i^click = { floor(t_i + τ_i^{delay,click}) if c_i = 1, and -1 otherwise }` |
| `∧`, `¬` | logical and, logical not | — | `∧`, `¬` |
| `~` before a quantity or a law | distributed as | — | `∼` |
| `~` before a number or a rough count | approximately | — | `≈` |
| `:=` | definition assignment | — | `:=` |
| ` x ` (ASCII), `·`, `×`, between two quantities in a product | multiplication | — | `·` |
| ` x ` (ASCII), `×`, between two index sets, or in a width-by-height size label | Cartesian product, and the size label | — | `×` |
| `[a, b, c]`, `{a, b, c}` | a set written out in full. A set literal takes braces, and a set literal written with its symbol is not set in code font | — | `{a, b, c}` |
| `²`, `^` | power | — | numeral superscript, `(·)²` |
| `≥`, `>=`, `≤`, `≡`, `≈`, `≠` | relations | — | `≥`, `≤`, `≡`, `≈`, `≠` |
| `→`, `->`, `↦` | tends to, maps to | — | `→`, `↦` |
| `∈` | set membership | — | `∈` |
| `∅` | empty set in a set expression | — | `{}` |
| NaN in a view | cell censored by `𝒞^k` | — | `∅` |
| NaN in the master (H4 on unsold rows) | the quantity does not exist on that row | H4 | `⊥` |
| `inf`, `∞` | infinity | — | `∞` |
| `\|·\|` | cardinality | — | `\|·\|` |
| i | auction row index | — | `i` |
| j | variable index within a row | — | `j` |
| k (subscript) | rival index | LR | `k` |
| k (superscript) | view index over C1-C4 | — | `k` |
| u, a, c | user, app, campaign pool indices | LU, LA, LC | `u`, `a`, `c` |
| d | day index | D1 | `d_i` |
| s, s(u) | observable segment index and map | — | `s(i)` |
| e, (e) | exchange index | B3 | `e_i` |
| e | Euler's number in gate(p) | LR3 | written `exp(·)` |
| -1 | observed no-event timestamp sentinel | E2, F2 | `-1` |
| {0,1} | binary domain | — | `{0,1}` |

### 2.12 Labels used as symbols

| Current | Meaning | Code | New |
|---|---|---|---|
| C1, C2, C3, C4 (unquoted) | the four experimental conditions. The backticked campaign codes are in 2.1 | — | `C1`-`C4`, view index `k` |
| #1 … #6 | the six modelled dependencies | — | unchanged |
| E1-E5 | rival-market graph edge labels | — | unchanged |
| G1-G4 (validation) | validation gate labels, renamed so they do not read as the LTV codes | — | `VG1`-`VG4` |
| G | genre taxonomy {casual, strategy, rpg, hypercasual} | C4 | `𝒢` |
| V-8 | edge-inventory entry reference | — | unchanged |
| subscript i | written in equations, dropped in prose | — | unchanged |
| always / funnel / ssp / master / none | observability statuses | — | words, unchanged |

### 2.13 Tables and records

| Current | Meaning | Code | New |
|---|---|---|---|
| master table, `auctions.parquet` | the complete uncensored table | — | `𝒟^full` |
| the C1-C4 views | the censored tables | — | `𝒟^{obs,k}` |
| one master row | per-row full record | — | `𝒟_i^full` |
| one view row | per-row observed record in view k | — | `𝒟_i^{obs,k}` |

---

## 3. Where each symbol comes from

Two lists. 3.1 is inherited from the paper, spelling and all, and each entry carries the paper line that has it. 3.2 is the quantities the paper has no symbol for, each formed by the rules in section 1. An inherited spelling is not by itself proof of an inherited meaning: check 3.1 for meaning as well as spelling, which is the check that catches `β_0^click`.

### 3.1 Inherited from the paper

| Symbol | Quantity | Paper line |
|---|---|---|
| `p_i^click`, `p_i^install` | true click and install probabilities | `main.tex:397-411` |
| `β_0^click`, `β_0^install` | funnel base levels. **Meaning differs**: the paper's are log-odds intercepts, this document's are multiplicative base rates on the probability scale (R5) | `main.tex:401-411` |
| `β_0(x_i)` | baseline rival price level | `main.tex:440` |
| `ω_k` | rival private value loading | `main.tex:440, 442` |
| `z_i^val` | standardised latent value index | `main.tex:440, 442` |
| `A_ik` | rival participation indicator | `main.tex:434, 436` |
| `B_ik` | rival k's bid on row i | `main.tex:440` |
| `g(·)` | logistic pacing map | `main.tex:436` |
| `η_i`, `v_i^proxy` | exploration draw, logging bid value | `main.tex:629` |
| `N_i` | realised competition density | `main.tex:521, 531`, `appendix.tex:28` |
| `𝒟^full`, `𝒟^{obs,k}` | master table and views | `main.tex:275, 279` |
| `ℓ̂^cat` | the hatted revenue form. **Meaning differs**: the paper's `ℓ̂(x_i)` is the fitted Tier-1 head, this document's is a generator-side category benchmark (2.4) | `main.tex:573, 575` |
| `F̂_m(b \| x_i)` | the paper's price-distribution model, the Tier-2 curve conditioned on the price | `main.tex:595` |
| `F_{u\|x}(b)`, `F̂_{u\|x}(b)` | the same Tier-2 curve, re-conditioned on the top rival bid rather than on the price, so it drops the floor. Equal to `F̂_m(b \| x_i)` for b ≥ f_i | `main.tex:595` |

### 3.2 Newly formed

| New symbol | Quantity |
|---|---|
| `t_i` | D1 timestamp |
| `e_i` | B3 exchange index |
| `d_i`, `d_i^week` | day index, week |
| `t_i^click`, `t_i^install` | E2 and F2 event timestamps |
| `τ_i^{delay,click}`, `τ_i^{delay,install}` | reporting delays |
| `t_0` | 28-day window origin |
| `y_i^pay` | G1 is_payer |
| `r_i^7d`, `r_i^30d` | G3 and G4 recognised revenue |
| `h_i` | market block of a row |
| `θ_i^click`, `θ_i^install`, `θ_i^pay` | LU2-LU4 latent propensities |
| `λ_u^ltv` | LU5 LTV multiplier |
| `χ_u^interest` | LU6 genre interest simplex |
| `χ_a^audience` | LA2 audience profile simplex |
| `λ_c^game` | LC2 game quality |
| `ω_0` | loading of the always-on rival (LR1) |
| `z_u^u`, `z_a^s`, `z_c^c`, `z_k^m`, `z^pool` | the four pool latent bundles |
| `A_i0` | participation of the always-on rival |
| `𝒜_i` | participant set |
| `w_i^riv` | lost by us and sold to a rival |
| `m_i^win` | the winning threshold max(u_i, f_i), the paper's body `m` |
| `γ^mkt`, `γ^riv` | OFF-path loading dial, value-awareness dial |
| `ρ^anchor` | anchored operating point 0.8 |
| `p_i^pay`, `p_i^{(stage)}` | true payer head, generic head over the stage family |
| `y_i^{(stage)}` | free funnel outcome in Part III |
| `ℓ_i` | true expected LTV given payer |
| `v_i^mkt` | supply-only EV channel |
| `s^u(i)`, `s^a(i)`, `s^c(i)`, `s^pair(i)`, `s^bin(i)`, `s^size(i)` | row-to-key maps |
| `z_i^{u,score}` | user value score |
| `ω_c^camp` | campaign draw weight |
| `α_i^genre` | genre relevance |
| `α_i^click`, `α_i^install`, `α_i^pay` | per-stage relevance multipliers |
| `α_i^{(stage)}`, `ω^{(stage)}` | family forms of the stage multiplier and stage weight, stage ∈ {click, install, pay} |
| `α_i^ease`, `α_i^time` | install ease, payer timing |
| `q_i^slot` | slot quality weight |
| `λ_i^os` | iOS spend multiplier |
| `μ^ltv(·)`, `μ_0^ltv`, `σ^ltv` | LTV lognormal location, intercept, scale |
| `β_0^pay` | payer base rate |
| `ω^click`, `ω^install`, `ω^pay` | stage weights |
| `δ^7d`, `δ^30d` | revenue recognition fractions |
| `ζ^shade`, `ζ^global`, `ζ^cpa` | own-bid scale dials |
| `σ^explore` | exploration spread |
| `ψ^floor`, `ψ^clearing` | price shape factors |
| `T^ecpm(·)`, `T^roas` | eCPM calibration target, reserved ROAS target |
| `μ^{delay,click}`, `μ^{delay,install}`, `σ^{delay,install}` | delay parameters |
| `σ^app`, `σ^creative`, `σ^game` | LA1, LC1, LC2 lognormal scales |
| `ξ^audience`, `ξ^interest` | Dirichlet concentrations |
| `n`, `n^user`, `n^cat`, `n^gaming`, `n^days` | counts |
| `μ^val`, `σ^val` | EV standardising constants |
| `n_0` | a realised rival-count value |
| `x_{1:n}`, `z_{1:n}^val` | all-row context and latent-value blocks |
| `χ^…` | 16 categorical share vectors, listed in 2.7 |
| `𝒟_i^full`, `𝒟_i^{obs,k}` | one row of the master and one row of a view |
| `𝒢`, `𝒫` | genre set, generator class |
| `∅`, `⊥` | censored cell, quantity that does not exist on the row |
| `VG1`-`VG4` | validation gate labels |

---

## 4. Where the symbols live

**What was counted.** Every Greek or decorated form the specification uses, in all the spellings that appear on disk. Unicode (`κ_u`, `ν^clk`, `σ_explore`, `ρ`), LaTeX braces (`\theta^U`, `κ_{u}`), and ASCII subscripts (`w_k`, `p_k`, `F_k`, `N_i`, `V_i`, `EV_i`, `z_i`, `v_i`, `r_i`, `m_u`, `b^own`, `y^win`, `y^clr`, `q^clk`). 76 distinct forms found. Counts are occurrences, not lines.

| | Files | Symbol hits | Code-identifier hits |
|---|---|---|---|
| Live | 29 | 421 | 674 |
| Superseded | 42 | 840 | 1,077 |
| **Carry symbols** | **71** | **1,261** | — |
| Code identifiers only, no symbols | 140 | 0 | 1,468 |

### 4.1 Live files

| File | Symbols | Code IDs | Note |
|---|---|---|---|
| `docs/v1/T9Sim_Specification_v10.md` | 179 | 165 | the specification |
| `Schema diagrams/KDD_Fig2_plate_dag.svg` | 43 | 3 | only live SVG carrying symbols |
| `t9_sim/src/t9sim/diagrams/make_fig2_plate_dag.py` | 42 | 4 | builds the SVG above |
| `docs/Simulator_Schema - June 10 (v7).md` | 26 | 80 | operative schema in the repo |
| `docs/Project_Design_v18.md` | 21 | 9 | 18 of 21 are bare `ρ` |
| `docs/v1/T9Sim_Implementation_Status.md` | 20 | 93 | spec companion |
| `t9_sim/src/t9sim/diagrams/make_param_map_v10_layers.py` | 14 | 36 | shared node map for the V10 set |
| `docs/v10_Anchor_Bands.md` | 11 | 0 | all `ρ` / `ρ*` |
| `docs/v1/v10_Training_Results.md` | 10 | 0 | all `ρ` |
| `docs/KDD_Schema_Decision_10Jul2026.md` | 9 | 0 | all `ρ` |
| `docs/PROJECT_LOG.md` | 8 | 27 | |
| `docs/v1/T9Sim_Config_Reference.md` | 5 | 141 | code-identifier heavy |
| `docs/T9Sim_DGP_Node_Register.md` | 5 | 131 | code-identifier heavy |
| `docs/Archive/T9Sim_Rebuild_Playbook.md` | 5 | 80 | |
| `docs/Archive/Klimis_Spec_Translator.md` | 4 | 21 | folded into the spec docx |
| `docs/KDD_TwoVersions_Comparison_12Jul2026.md` | 3 | 0 | |
| `t9_sim/src/t9sim/diagrams/make_bn_anchored.py` | 2 | 29 | both hits are dict keys, see 4.3 |
| `docs/Method_Benchmark_10M_Results_13Jul2026.md` | 2 | 0 | |
| `docs/Method_Benchmark_1M_Results_13Jul2026.md` | 2 | 0 | |
| `Schema diagrams/Variable formulas/H2_bid_price.svg` | 1 | 10 | `σ_explore` |
| `t9_sim/src/t9sim/diagrams/make_outcome_svgs.py` | 1 | 33 | builds `Variable formulas/` |
| `t9_sim/src/t9sim/diagrams/make_fig3_two_dials.py` | 1 | 26 | |
| `t9_sim/src/t9sim/diagrams/make_source_matrix.py` | 1 | 13 | |
| `docs/T9Sim_Appendix_AD.md` | 1 | 15 | |
| `docs/SHAP_Analysis_v10.md` | 1 | 7 | |
| `docs/Schema_Review_Reminders.md` | 1 | 2 | |
| `docs/DeepRead_ESMM_MAL_MAC.md` | 1 | 2 | |
| `docs/Citation_Register.md` | 1 | 0 | |
| `docs/results/sweep_wk_v10_100k.json` | 1 | 2 | `w_k` in a metadata string |

Borderline, listed for completeness. `Schema diagrams/Variable formulas/G2_ltv_value.svg` has 2 bare `σ` in `LogNormal(μ_cat, σ)`. The specification writes G2 with `mu_cat` and a literal 1.648169, so bare `σ` is the diagram's own label and is not in the 2.7 key.

The live Schema V10 map set carries no symbols. `Schema V10 - All Layers.svg`, `V10a - Generation.svg`, `V10b - Dependencies.svg`, `V10c - private rival prices.svg` and `Schema V10 - Overview.svg` all measure 0. They were already relabelled to plain English (`make_param_map_v10_layers.py:154-157`). Only the Archive copies still carry symbols.

### 4.2 Superseded files

| File | Symbols | Code IDs | Why superseded |
|---|---|---|---|
| `docs/Archive/T9Sim_Symbol_Reconciliation.md` | 114 | 68 | earlier attempt at this job |
| `docs/Archive/Notation_Canonical.md` | 99 | 51 | earlier attempt at this job |
| `docs/BN_Formal_Definition_Response.md` | 97 | 35 | formalisation trail, folded into the spec |
| `docs/BN_Formalisation_Readable.md` | 58 | 8 | formalisation trail |
| `docs/SSP_Null_Validity_Discussion_5Jul2026.md` | 46 | 21 | dated record, 35 of 46 are `ρ` |
| `docs/KDD_Paper_Draft_v2.md` | 43 | 29 | paper now in `release/Paper/Overleaf/*.tex` |
| `docs/Simulator_Schema - June 10 (v8).md` | 38 | 102 | proposed, never merged |
| `docs/Schema_v10_Clarity_Review_21Jul2026.md` | 26 | 6 | dated review |
| `docs/Archive/KDD_8Pager_Structure_Proposal_10Jul2026.md` | 19 | 2 | archived |
| `docs/KDD_Structure_RedTeam_Critique_11Jul2026.md` | 18 | 1 | dated critique |
| `Schema diagrams/Archive/Full_parameter_map_v10.svg` | 16 | 26 | archived |
| `docs/Formalisation_Compliance_Review_14Jul2026.md` | 15 | 65 | dated review |
| `t9_sim/.../make_bn_formal_dag.py` | 15 | 26 | output `T9_BN_Formal_DAG.svg` no longer exists |
| `t9_sim/.../make_param_map_v11.py` | 15 | 24 | v11 arms abandoned |
| `Schema diagrams/Archive/Full_parameter_map_v11.svg` | 15 | 22 | archived |
| `docs/Archive/Draft_KDD_submission_Ken_v2.md` | 15 | 1 | archived |
| `t9_sim/.../make_bn_formal_dag_v2.py` | 14 | 31 | output no longer exists |
| `t9_sim/.../make_param_map_v10_split.py` | 14 | 31 | output archived |
| `t9_sim/.../make_param_map_v11_split.py` | 14 | 27 | v11 abandoned |
| `Schema diagrams/Archive/Full_parameter_map_v10_split.svg` | 14 | 24 | archived |
| `Schema diagrams/Archive/Full_parameter_map_v11_split.svg` | 14 | 24 | archived |
| `docs/Archive/Draft_KDD_submission_ProposalStructure.md` | 14 | 1 | archived |
| `Schema diagrams/Archive/Schema V10c - private rival prices.svg` | 13 | 25 | archived |
| `Schema diagrams/Archive/Schema V10a - Generation.svg` | 13 | 20 | archived |
| `Schema diagrams/Archive/Schema V10b - BN.svg` | 13 | 20 | archived |
| `t9_sim/.../make_param_map_v11_abstract.py` | 10 | 26 | v11 abandoned |
| `Schema diagrams/Archive/Full_parameter_map_v11_abstract.svg` | 10 | 22 | archived |
| `Schema diagrams/Archive/Full_parameter_map_v11_changes.svg` | 9 | 21 | archived |
| `docs/Archive/Session_Summary_v11-arms-KDD-decision_11Jul2026.md` | 7 | 1 | archived |
| `docs/Archive/Notation_Reconciliation_Report.md` | 6 | 4 | earlier attempt at this job |
| `docs/Archive/KDD_Paper_Draft.md` | 5 | 30 | archived |
| `docs/KDD_Metric_Selection_19Jul2026.md` | 4 | 8 | dated note |
| `Schema diagrams/Archive/Full_parameter_map_v9.svg` | 3 | 18 | archived |
| `docs/BN_Formalisation_Critique_20Jul2026.md` | 3 | 7 | dated critique |
| `t9_sim/.../make_schema_overview_v10b.py` | 2 | 5 | output archived |
| `Schema diagrams/Archive/Full_parameter_map_v8.svg` | 2 | 19 | archived |
| `Schema diagrams/Archive/Full_parameter_map_v8 - Copy.svg` | 2 | 19 | archived |
| `t9_sim/.../make_schema_overview.py` | 1 | 3 | output archived |
| `Schema diagrams/Archive/Schema_Overview_v10.svg` | 1 | 3 | archived |
| `Schema diagrams/Archive/Schema_Overview_v10B#.svg` | 1 | 3 | archived |
| `docs/BN_Design_Investigation.md` | 1 | 14 | formalisation trail |
| `docs/KDD_Sections_4-8_Work.md` | 1 | 0 | superseded by the LaTeX |

### 4.3 Code identifiers, counted separately

3,219 hits across 211 files in the same 3 directories are code identifiers, not symbols. They are parquet columns, YAML keys or Python variables and none of them changes.

The set that sits inside the specification's laws and its 2.7 key is `v_slot`, `ease`, `t_pay`, `plat`, `base_ctr`, `base_ir`, `base_payer`, `m_click`, `m_install`, `m_pay`, `la1`, `lc1`, `lc2`, `e_ltv`, `ev_truth`, `shade`, `k_global`, `k_cpa`, `base_e`, `w_k`, `sigma_k`, `pi_ke`, `beta_R`, `phi_p`, `rho`, `sigma_explore`, `pacing_ar`, `pacing_sigma`, `lc2_game_quality`, `lu7_competing_bid`, `bid_density`, `winning_price`, `floor_price`, `bid_price`, `p_click`, `p_install`, `p_payer`, `n_gaming`.

Added 16 August 2026, when the node-register review gave nine settings a key for the first time and
the validator read their bare names as notation: the four pool sizes `n_users`, `n_apps`,
`n_campaigns` and `n_advertisers`, plus `pool_sizes`, `advertiser_share`, `campaign_share`,
`ad_exchanges`, `slot_format_shares`, `slot_quality`, `size_given_format` and `k_audience`. Every one
is a YAML key. The `n_` group is the reason this paragraph exists rather than a rule change: a
subscript is an index in the notation and a word in a config key, and only the census can tell them
apart.

| Fact | Detail |
|---|---|
| 1 form is both a symbol and a live code identifier | `w_k`, 117 hits across 41 files, and also `RivalPool.w_k` at `t9_sim/src/t9sim/rival_pool.py:55`. |
| Greek never collides | The code spells its versions out. Symbol `σ_k` vs code `sigma_k` (`rival_pool.py:63`), symbol `π_k` vs code `pi_ke` (`:66`), symbol `ρ` vs config key `rho`. |

`make_bn_anchored.py` is the clearest example of the distinction. Both its `w_k` hits are Python dict keys used as graph node ids (`:63`, `:118`), and the label that actually renders is `RivalValueLoading`. 2 hits, 2 code identifiers, 0 symbols.

### 4.4 Excluded

5 files matched on Greek but carry a cited paper's notation, not the specification's. They are not in the counts above.

| File | Hits | What it actually is |
|---|---|---|
| `docs/Archive/KDD 9 papers consolidated.md` | 11 | `κ, σ, θ` from a quoted algorithm |
| `docs/DeepRead_WinningPrice_Top5.md` | 4 | `ℓ` as a win/loss indicator |
| `docs/DeepRead_LTV_Sources.md` | 3 | Spearman `ρ`, ZILN `σ_i` |
| `docs/DeepRead_Auction5_WinningPrice.md` | 1 | `v_i` from Wu et al. |
| `docs/T9_LitSweep_RTB_Auctions_2Jul2026_records.json` | 1 | `v_i` from an abstract |
