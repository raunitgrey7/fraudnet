# Benchmark results

Seed 42, full profile, generated 2026-11-27T06:49:31Z, commit `2e2700b`, Python 3.13.17, torch 2.14.1, scikit-learn 1.9.1 on arm64 with 2 threads. Total benchmark time 339.4 s.

## Dataset

441,378 transactions, 5,076 accounts, 8,039 devices, 406 merchants; 2,982 fraud (0.676 percent); 14.0 percent of fraud is never reported; median label delay 9.99 days.

| window | days | transactions | fraud | fraud rate | fraud not yet labelled at build |
|---|---|---|---|---|---|
| train | 7 to 49 | 200,586 | 1347 | 0.671% | 35.3% |
| val | 49 to 63 | 69,456 | 462 | 0.665% | 54.8% |
| test | 70 to 90 | 104,125 | 786 | 0.755% | n/a |

## Detection on the test window

| model | PR-AUC [95% CI] | ROC-AUC | recall at 0.5% FPR | recall at 1% FPR [95% CI] | precision top-10/day | precision top-50/day |
|---|---|---|---|---|---|---|
| rules | 0.4829 [0.381, 0.571] | 0.8311 | 0.4847 | 0.5038 [0.399, 0.588] | 0.815 | 0.370 |
| gbdt_tabular | 0.7175 [0.643, 0.775] | 0.9929 | 0.7850 | 0.8677 [0.809, 0.911] | 0.805 | 0.521 |
| gbdt_streaming | 0.8173 [0.757, 0.886] | 0.9939 | 0.8410 | 0.8893 [0.810, 0.962] | 0.945 | 0.572 |
| gbdt | 0.8772 [0.843, 0.911] | 0.9985 | 0.9249 | 0.9733 [0.952, 0.986] | 0.960 | 0.590 |
| graphsage | 0.7204 [0.635, 0.801] | 0.9865 | 0.7036 | 0.7506 [0.639, 0.860] | 0.950 | 0.513 |
| gbdt+graphsage | 0.8044 [0.749, 0.860] | 0.9966 | 0.7735 | 0.8740 [0.819, 0.934] | 0.960 | 0.548 |

## Money saved under the cost model

| model | alerts/day | alert precision | fraud recall | money saved | share of loss saved | precision at 50/day | recall at 50/day | money saved at 50/day |
|---|---|---|---|---|---|---|---|---|
| rules | 33.9 | 0.551 | 0.475 | 163,350 | 0.613 | 0.370 | 0.471 | 156,709 |
| gbdt_tabular | 38.5 | 0.681 | 0.668 | 216,174 | 0.811 | 0.521 | 0.663 | 209,311 |
| gbdt_streaming | 25.4 | 0.911 | 0.588 | 195,182 | 0.732 | 0.572 | 0.728 | 213,694 |
| gbdt | 27.0 | 0.913 | 0.627 | 199,148 | 0.747 | 0.590 | 0.751 | 214,609 |
| graphsage | 20.6 | 0.934 | 0.488 | 180,088 | 0.675 | 0.513 | 0.653 | 194,614 |
| gbdt+graphsage | 22.9 | 0.948 | 0.551 | 182,069 | 0.683 | 0.548 | 0.697 | 203,151 |

Cost model: 5 per alert reviewed, 10 more when the alert is a genuine payment, a missed fraud costs its amount plus a 15 chargeback fee. The first block uses the threshold that maximised money saved on validation; the second gives analysts exactly 50 alerts per day. Fraud loss with no model: 266,587; a perfect detector would save 262,657.

## Per-pattern recall at 1 percent FPR

| model | fraud_ring | money_mule | account_takeover | card_testing |
|---|---|---|---|---|
| rules | 0.372 / 0.80 | 0.729 / 1.00 | 0.989 / 1.00 | 0.271 / 1.00 |
| gbdt_tabular | 0.794 / 1.00 | 0.971 / 1.00 | 0.859 / 1.00 | 0.950 / 1.00 |
| gbdt_streaming | 0.784 / 1.00 | 0.982 / 1.00 | 1.000 / 1.00 | 0.993 / 1.00 |
| gbdt | 0.948 / 1.00 | 0.994 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 |
| graphsage | 0.505 / 0.80 | 0.977 / 1.00 | 1.000 / 1.00 | 0.986 / 1.00 |
| gbdt+graphsage | 0.750 / 1.00 | 0.988 / 1.00 | 1.000 / 1.00 | 0.993 / 1.00 |

Cells: transaction recall / share of attacks with at least one transaction caught. Test window: fraud_ring 384 transactions in 5 attacks, money_mule 170 transactions in 36 attacks, account_takeover 92 transactions in 26 attacks, card_testing 140 transactions in 6 attacks.

## Graph feature ablation (gradient boosting)

| variant | PR-AUC | change vs tabular [paired 95% CI] | recall at 1% FPR | fraud_ring | money_mule | account_takeover | card_testing |
|---|---|---|---|---|---|---|---|
| tabular | 0.7175 | +0.0000 [+0.000, +0.000] | 0.8677 | 0.794 | 0.971 | 0.859 | 0.950 |
| tabular+shared_device | 0.8146 | +0.0971 [+0.061, +0.138] | 0.9262 | 0.878 | 0.965 | 0.946 | 1.000 |
| tabular+counterparty | 0.7377 | +0.0202 [+0.009, +0.033] | 0.8779 | 0.802 | 0.982 | 0.870 | 0.964 |
| tabular+risk | 0.8239 | +0.1064 [+0.074, +0.148] | 0.9288 | 0.867 | 0.988 | 0.989 | 0.986 |
| tabular+degree | 0.8269 | +0.1094 [+0.084, +0.140] | 0.9288 | 0.885 | 0.977 | 0.924 | 0.993 |
| tabular+pagerank | 0.8334 | +0.1159 [+0.079, +0.170] | 0.9402 | 0.901 | 0.988 | 0.924 | 1.000 |
| tabular+community | 0.8069 | +0.0894 [+0.053, +0.131] | 0.9351 | 0.922 | 0.977 | 0.859 | 0.971 |
| all-shared_device | 0.8862 | +0.1687 [+0.126, +0.232] | 0.9746 | 0.953 | 0.988 | 1.000 | 1.000 |
| all-counterparty | 0.8794 | +0.1619 [+0.118, +0.221] | 0.9707 | 0.945 | 0.988 | 1.000 | 1.000 |
| all-risk | 0.7877 | +0.0702 [+0.039, +0.118] | 0.9173 | 0.865 | 0.977 | 0.902 | 1.000 |
| all-degree | 0.8702 | +0.1527 [+0.110, +0.209] | 0.9784 | 0.956 | 1.000 | 1.000 | 1.000 |
| all-pagerank | 0.8715 | +0.1540 [+0.115, +0.212] | 0.9656 | 0.935 | 0.988 | 1.000 | 1.000 |
| all-community | 0.8300 | +0.1125 [+0.060, +0.192] | 0.9211 | 0.841 | 0.994 | 1.000 | 1.000 |
| all | 0.8772 | +0.1597 [+0.117, +0.218] | 0.9733 | 0.948 | 0.994 | 1.000 | 1.000 |

Pattern columns are transaction recall at 1 percent FPR. Intervals resample test days and score both models on the same days.

## Label-maturity gap

| model | protocol | PR-AUC | recall at 1% FPR | ring recall | money saved |
|---|---|---|---|---|---|
| gbdt | with_gap | 0.8772 | 0.9733 | 0.948 | 199,148 |
| gbdt | no_gap | 0.7843 | 0.9631 | 0.938 | 161,076 |
| graphsage | with_gap | 0.7204 | 0.7506 | 0.505 | 180,088 |
| graphsage | no_gap | 0.7890 | 0.8435 | 0.685 | 193,207 |

Without the gap, validation runs from day 56 to the build day and 67.7 percent of its fraud is unlabelled at build time.

## Point-in-time correctness

First 30 days (139,394 transactions), cut at day 21.29.

| pipeline | check | rows | values compared | mismatched values | result |
|---|---|---|---|---|---|
| honest | prefix_replay | 97161 | 5,149,533 | 0 | identical |
| honest | scrambled_future | 97161 | 5,149,533 | 0 | identical |
| leaky_instant_labels | prefix_replay | 97161 | 5,149,533 | 55,160 | differs (device_known_fraud, acct_known_fraud, cp_known_fraud, acct_devices_known_fraud) |
| leaky_instant_labels | scrambled_future | 97161 | 5,149,533 | 115,207 | differs (device_known_fraud, acct_known_fraud, cp_known_fraud, acct_devices_known_fraud) |

## Cost of a label leak

| setting | test PR-AUC | recall at 1% FPR |
|---|---|---|
| leaky_model_on_leaky_features | 0.9226 | 0.9949 |
| leaky_model_on_online_features | 0.5472 | 0.5394 |
| honest_model_on_online_features | 0.8772 | 0.9733 |

## Scoring latency

| path | p50 ms | p95 ms | p99 ms | max ms | sequential txn/s |
|---|---|---|---|---|---|
| rules | 0.068 | 0.074 | 0.086 | 0.131 | 14673.0 |
| rules_no_reasons | 0.039 | 0.043 | 0.048 | 0.079 | 25654.0 |
| gbdt | 1.155 | 1.232 | 1.351 | 63.750 | 815.4 |
| gbdt_no_reasons | 1.113 | 1.264 | 1.423 | 13.041 | 878.4 |
| graphsage | 0.065 | 0.073 | 0.090 | 0.157 | 15072.2 |
| graphsage_no_reasons | 0.046 | 0.051 | 0.061 | 0.125 | 21362.7 |
| gbdt over HTTP, server side | 1.444 | 1.604 | 1.704 | 2.037 | 684.1 |
| gbdt over HTTP, client side | 1.949 | 2.102 | 2.212 | 2.514 | 509.4 |

2000 test transactions scored one at a time after 50 untimed warm-up transactions; each path covers feature lookup, scoring and reason codes unless marked no_reasons. Online features equal the offline replay features (max absolute difference 0.0). A daily graph refresh over 12,144 nodes takes 0.233 s plus 0.007 s of GNN embedding.

## Seeds

| seed | rules | gbdt_tabular | gbdt | graphsage |
|---|---|---|---|---|
| 42 | 0.4829 | 0.7175 | 0.8772 | 0.7204 |
| 7 | 0.5890 | 0.8178 | 0.9370 | 0.9586 |
| 2026 | 0.5237 | 0.6525 | 0.9071 | 0.8974 |
| mean | 0.5319 | 0.7293 | 0.9071 | 0.8588 |
| standard deviation | 0.0535 | 0.0833 | 0.0299 | 0.1237 |

Test PR-AUC per simulator seed; each seed is a different world with the same configuration.
