# Evaluation

How the benchmark is built, what each metric means, and how to reproduce every number. The numbers
themselves live in `benchmarks/results.json` (and the tables generated from it in the README and
`benchmarks/results.md`).

## Reproduce

```
make setup                 # uv sync; simulate and cache the seed-42 world
make bench                 # python -m fraudnet.bench --seed 42 --seeds 7,2026 --update-readme --check
make bench-quick           # the CI profile (smaller world) with its own thresholds
make tune                  # the validation-only hyperparameter sweep
make leakage               # the point-in-time checker on its own
```

The run is deterministic for a given seed on a given platform: the simulator uses one seeded generator in a
fixed order, boosting is deterministic, and GNN training is pinned to one thread because two threads made
its sparse backward passes reduce in a varying order (REPORT.md has the measurement). `tests/test_sim.py`
and `tests/test_models.py` check that the same seed gives byte-identical data and identical predictions.

## Dataset

A seeded agent-based simulation (`fraudnet.sim`) of 5,000 customers, their devices, 400 merchants and 90 days
of card purchases and account-to-account transfers.

**Genuine behaviour.** Each customer has a log-normal purchase rate and spend level (heavy-tailed, as card
spend is), a short list of favourite merchants drawn from category preferences, a merchant popularity that
follows a Zipf law within each category, an hour-of-day profile (a few customers are night owls), a weekday
pattern, one or two personal devices and possibly a phone change during the window. Customers transfer money
to social contacts that are mostly local (neighbourhood blocks), which gives the transfer graph realistic
clustering.

**Planted fraud** (ground truth per transaction, with an attack id):

| Pattern | Mechanism | Camouflage | Report delay |
|---|---|---|---|
| Fraud ring | 4 to 8 synthetic identities sign up within 10 days, share 2 to 4 devices, warm up for 2 to 4 weeks with ordinary purchases (sometimes paying each other), then bust out with large purchases, half of the rings at a colluding merchant | Warm-up spend is sized and placed like a new customer's | Discovered 15 to 40 days after the bust-out |
| Money mule chain | A scam victim's transfer passes through 2 to 4 recruited mules; each keeps 3 to 20 percent and forwards within minutes to hours; exit through crypto or gift cards or an exit account | Mules have normal lives; some hops reuse a payee; some hops from a herder's device | Victim reports in about 1.5 days; hops traced 0.5 to 8 days later |
| Account takeover | Smash-and-grab (fast night burst of high-value purchases) or low-and-slow (a few moderately larger purchases over a day at ordinary hours); attacker devices are reused across victims; a quarter push money into a mule chain | Low-and-slow amounts are 1.5 to 4 times typical spend | Victim notices in about 2 days |
| Card testing | Tiny charges on 10 to 32 stolen cards at popular digital merchants, as a fast burst from one device or a slow drip across a farm of 6 to 12 devices at app-store prices; a quarter of cards cashed out later | Drip charges use real app prices and spread across three merchants | 55 percent of tests never reported; the rest after 5 to 35 days |

**Genuine look-alikes**, one for each pattern: shared household devices (2 to 4 accounts) and workplace
computers (5 to 12 colleagues); trips and shopping sprees (bursts of larger purchases at new merchants, at odd
hours); in-app purchase bursts (several small charges at a digital merchant within an hour); landlords, rent
collection, forwarding and large one-off transfers (money in from several accounts and out again, big
payments to first-time payees); group payments.

On top of the pattern-specific delays, 3 percent of fraud is never reported. The fraud rate is under 1 percent
of transactions (the dataset table in the README has the measured value).

## Protocol

* **Burn-in** (days 0 to 6): warms the feature store; never fitted or scored.
* **Train** (days 7 to 48) and **validation** (days 49 to 62): labels are the reports that arrived by the
  build day (day 70). Fraud reported later is treated as genuine, exactly as a bank retraining on day 70
  would see it. The README dataset table shows how much fraud is still unlabelled at build time.
* **Label-maturity gap** (days 63 to 69): processed by the feature store, excluded from fitting, because most
  of its fraud is not yet reported. The benchmark also reruns the main models without the gap (validation up
  to day 70) and reports both.
* **Test** (days 70 to 89): scored with features computed from the log only, evaluated against the full
  ground truth.

The features are the same in every protocol, because they depend only on time, not on the split.

Model selection uses validation only: the boosted model's tree count (validation average precision every 20
trees), the GNN's epoch (validation average precision with patience), the decision thresholds, and the
hyperparameter sweep in `fraudnet.tune`. Test labels are read only by the evaluation.

## Metrics

All computed on the test window with `evaluation/metrics.py`.

* **PR-AUC**: average precision (scikit-learn `average_precision_score`). With 0.7 percent positives it is far
  more informative than ROC-AUC, which is reported for completeness.
* **Recall at x percent FPR**: the share of fraud scored strictly above the threshold that flags at most x
  percent of genuine test transactions. Ties at the threshold are not flagged, so ties can only lower recall.
* **Precision at top-k per day**: for each test day, the fraud share among the k highest-scored
  transactions, averaged over days (k = 10 and 50): what an analyst team with a fixed daily capacity sees.
* **Money saved**: under the cost model below, the fraud loss avoided minus the cost of the alerts, at the
  threshold that maximised money saved on validation (deployable), and with a fixed budget of the top 50
  alerts per day. Also reported: the loss with no model and what a perfect detector would save.
* **Per-pattern recall**: transaction recall of each planted pattern at the 1 percent FPR threshold, and the
  share of attacks with at least one transaction caught.
* **Confidence intervals**: 95 percent percentile intervals from 200 day-block bootstrap resamples (whole test
  days drawn with replacement), because fraud arrives in attacks and transactions within a day are not
  independent. Ablation intervals are paired: both models are scored on the same resampled days and the
  interval is of the difference.

### Cost model

| Event | Cost |
|---|---|
| Any alert | 5 (analyst review) |
| Alert on a genuine payment | 10 more (customer friction: the payment is held or declined) |
| Missed fraud | its amount plus a 15 chargeback fee |
| Caught fraud | only the review |

Money saved = loss with no model minus (missed fraud cost plus alert costs).

## Experiments

* **Main models**: rules, boosting on tabular features, boosting on tabular plus streaming graph features,
  boosting on all features, GraphSAGE, and the mean of boosting and GraphSAGE.
* **Graph feature ablation**: boosting on the tabular base plus one graph group at a time, and on all features
  minus one group at a time.
* **Label-maturity gap**: boosting and GraphSAGE refitted without the gap.
* **Point-in-time proof**: `check_point_in_time` on the first 30 days of the real dataset with a cut at day
  21.29, for the honest pipeline and for a pipeline that makes every eventual report visible at transaction
  time. Two variants: replay of the log as known at the cut (earlier transactions, reports that arrived
  before the cut), and a scrambled future (amounts, devices and merchants after the cut redrawn, fake reports
  after the cut for genuine past transactions). Every feature of every transaction before the cut must be
  bit-identical to the full run.
* **Cost of a label leak**: boosting trained on leaky features, evaluated on leaky test features (what a
  careless offline evaluation reports) and on the features the API can compute (what production sees).
* **Latency**: 2,000 test transactions scored one at a time through `ScoringService` (feature lookup, score,
  reason codes; also without reason codes) after 50 untimed warm-up transactions, for every model, from the
  same pickled build-day store; 500 requests through the FastAPI app in process (server time from
  `X-Process-Time-ms`, client time around the call); the time of one daily graph refresh. The same loop checks
  that online features equal the offline replay features.
* **Seeds**: rules, tabular boosting, full boosting and GraphSAGE on worlds simulated with seeds 7 and 2026.

## Quality gate

`benchmarks/thresholds.json` holds bounds for the full and the quick profile; `python -m fraudnet.bench
--check` fails when a result breaches one. CI runs the quick profile and re-checks the committed full results.
