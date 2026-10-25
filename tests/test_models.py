"""Rules, gradient boosting, the GraphSAGE model and the reason-code explainer."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

from fraudnet.features.replay import GnnSnapshot
from fraudnet.features.schema import FEATURE_INDEX, GROUPS, N_FEATURES, TABULAR_GROUPS, columns_for
from fraudnet.models.explain import OcclusionExplainer
from fraudnet.models.gbdt import GBDTModel
from fraudnet.models.gnn import (
    GNNConfig,
    GNNModel,
    HeteroSAGELayer,
    TabularEncoder,
    resolve_nodes,
    snapshot_tensors,
)
from fraudnet.models.rules import RULES, RulesModel
from fraudnet.pipeline import fit_gbdt, prepare
from fraudnet.sim import SimConfig
from fraudnet.sim.config import TRANSFER_CATEGORY


def _test_ap(prepared, score: np.ndarray) -> float:
    return float(average_precision_score(prepared.labels["test"][prepared.masks["test"]], score))


def _base_rate(prepared) -> float:
    return float(prepared.labels["test"][prepared.masks["test"]].mean())


def test_rules_fire_on_a_known_fraud_device():
    row = np.zeros(N_FEATURES, dtype=np.float32)
    row[FEATURE_INDEX["device_known_fraud"]] = np.log1p(2)
    row[FEATURE_INDEX["device_accts_24h"]] = 4
    reasons = RulesModel().reasons(row)
    assert [r["code"] for r in reasons] == ["KNOWN_FRAUD_DEVICE", "DEVICE_FANOUT"]
    assert RulesModel().score(row[None])[0] > RulesModel().score(np.zeros((1, N_FEATURES)))[0] + 6.9


def test_rules_have_unique_codes_and_positive_weights():
    codes = [r.code for r in RULES]
    assert len(codes) == len(set(codes))
    assert all(r.weight > 0 for r in RULES)


def test_rules_beat_chance_on_the_small_world(prepared):
    score = RulesModel().score(prepared.table.X[prepared.masks["test"]])
    assert _test_ap(prepared, score) > 3 * _base_rate(prepared)


def test_boosted_model_beats_rules_and_chooses_its_size_on_validation(bundle, prepared):
    X_test = prepared.table.X[prepared.masks["test"]]
    gbdt_ap = _test_ap(prepared, bundle.gbdt.predict(X_test))
    assert gbdt_ap > _test_ap(prepared, RulesModel().score(X_test))
    assert bundle.gbdt.n_iter == max(bundle.gbdt.val_curve, key=lambda p: p[1])[0]
    assert bundle.gbdt.clf.n_iter_ == bundle.gbdt.n_iter, "the deployed model is refitted at the chosen size"


def test_refitted_trees_match_the_staged_prediction(prepared):
    cols = columns_for(TABULAR_GROUPS)
    staged = fit_gbdt(prepared, cols, "a")
    refit = fit_gbdt(prepared, cols, "b", refit=True)
    X_test = prepared.table.X[prepared.masks["test"]]
    assert staged.n_iter == refit.n_iter
    assert np.allclose(staged.predict(X_test), refit.predict(X_test), atol=1e-12)


def test_boosted_model_requires_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        GBDTModel([0, 1]).predict(np.zeros((1, N_FEATURES)))


def test_gbdt_probabilities_are_valid(bundle, prepared):
    p = bundle.gbdt.predict(prepared.table.X[prepared.masks["test"]])
    assert np.all((p >= 0) & (p <= 1)) and np.isfinite(p).all()


def test_snapshot_tensors_are_row_normalised(prepared):
    snap = prepared.table.snapshots[-1]
    st = snapshot_tensors(snap)
    assert st.x.shape[0] == st.n_nodes + 3
    for adj in st.adjs:
        sums = torch.sparse.sum(adj.to_sparse_coo(), dim=1).to_dense()
        nonzero = sums[sums > 0]
        assert torch.allclose(nonzero, torch.ones_like(nonzero), atol=1e-5)
    assert torch.equal(st.x[-3:, 3], torch.ones(3)), "unknown-entity nodes carry the unknown flag"


def test_unknown_entities_map_to_their_own_nodes():
    X = np.zeros((2, N_FEATURES), dtype=np.float32)
    X[1, FEATURE_INDEX["merchant_category"]] = TRANSFER_CATEGORY
    nodes = np.array([[-1, -1, -1], [3, 4, -1]])
    out = resolve_nodes(nodes, X, 10)
    assert out.tolist() == [[10, 11, 12], [3, 4, 10]]


def test_sage_layer_shape_and_mean_aggregation():
    layer = HeteroSAGELayer(4, 6, n_relations=1)
    h = torch.randn(3, 4)
    adj = torch.sparse_coo_tensor(
        torch.tensor([[0, 0], [1, 2]]), torch.tensor([0.5, 0.5]), (3, 3), check_invariants=True
    )
    adj = adj.to_sparse_csr()
    assert layer(h, [adj]).shape == (3, 6)


def test_tabular_encoder_is_finite_with_one_hot_codes(prepared):
    enc = TabularEncoder().fit(prepared.table.X[prepared.masks["train"]])
    out = enc.transform(prepared.table.X[:100])
    assert out.shape == (100, enc.dim) and np.isfinite(out).all()
    assert np.allclose(out[:, -14:].sum(axis=1), 2.0), "one channel and one merchant code per row"


def test_gnn_learns_and_is_deterministic(prepared):
    config = GNNConfig(epochs=2, patience=2)
    a = GNNModel(config).fit(
        prepared.table, prepared.labels["train"], prepared.masks["train"], prepared.masks["val"]
    )
    b = GNNModel(config).fit(
        prepared.table, prepared.labels["train"], prepared.masks["train"], prepared.masks["val"]
    )
    test = prepared.masks["test"]
    pa, pb = a.predict(prepared.table, test), b.predict(prepared.table, test)
    assert np.array_equal(pa, pb)
    assert _test_ap(prepared, pa) > 3 * _base_rate(prepared)


@pytest.mark.slow
def test_gnn_is_bit_identical_where_torch_runs_in_parallel():
    """The quick world is large enough for torch's parallel kernels; the default config must still repeat."""
    prep = prepare(SimConfig.quick(), None)
    runs = [
        GNNModel(GNNConfig(epochs=2, patience=2)).fit(
            prep.table, prep.labels["train"], prep.masks["train"], prep.masks["val"]
        )
        for _ in range(2)
    ]
    test = prep.masks["test"]
    assert np.array_equal(runs[0].predict(prep.table, test), runs[1].predict(prep.table, test))


def test_gnn_online_scores_match_batch_scores(bundle, prepared):
    test_rows = np.flatnonzero(prepared.masks["test"])[:200]
    mask = np.zeros(prepared.table.X.shape[0], dtype=bool)
    mask[test_rows] = True
    batch = bundle.gnn.predict(prepared.table, mask)
    sid = int(prepared.table.snap_id[test_rows[0]])
    same_snapshot = prepared.table.snap_id[test_rows] == sid
    h, n = bundle.gnn.embed_snapshot(prepared.table.snapshots[sid])
    online = bundle.gnn.score_rows(
        h, n, prepared.table.nodes[test_rows][same_snapshot], prepared.table.X[test_rows][same_snapshot]
    )
    assert np.allclose(online, batch[same_snapshot], atol=1e-6)
    assert isinstance(prepared.table.snapshots[sid], GnnSnapshot)


def test_explainer_ranks_the_group_that_drives_the_score():
    baseline = np.zeros(N_FEATURES, dtype=np.float32)
    explainer = OcclusionExplainer(baseline)
    row = np.zeros(N_FEATURES, dtype=np.float32)
    row[FEATURE_INDEX["device_accts_total"]] = 9.0
    row[FEATURE_INDEX["acct_cnt_1h"]] = 2.0

    def predict(batch: np.ndarray) -> np.ndarray:
        return (
            0.08 * batch[:, FEATURE_INDEX["device_accts_total"]]
            + 0.01 * batch[:, FEATURE_INDEX["acct_cnt_1h"]]
        )

    batch = explainer.variants(row)
    assert batch.shape == (1 + len(GROUPS), N_FEATURES)
    reasons = explainer.reasons(row, predict(batch))
    assert [r.code for r in reasons] == ["SHARED_DEVICE", "VELOCITY_SPIKE"]
    assert reasons[0].contribution == pytest.approx(0.72, abs=1e-5)
    assert "9 accounts" in reasons[0].detail


def test_explainer_gives_no_reasons_for_a_flat_model():
    explainer = OcclusionExplainer(np.zeros(N_FEATURES, dtype=np.float32))
    row = np.ones(N_FEATURES, dtype=np.float32)
    assert explainer.reasons(row, np.full(1 + len(GROUPS), 0.3)) == []
