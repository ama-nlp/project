"""E7 checkpoint-difference math, checked against dense products on small matrices."""

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import interp_ckpt_diff as cd  # noqa: E402


def factors(rng, d_out=30, d_in=20, r=3):
    return rng.standard_normal((r, d_in)), rng.standard_normal((d_out, r))


def test_change_stats_match_dense():
    rng = np.random.default_rng(0)
    f = {n: factors(rng) for n in ("s90", "s100", "s110", "s120", "s400")}
    w = {n: b @ a for n, (a, b) in f.items()}
    order = ["s90", "s100", "s110", "s120"]
    st = cd.change_stats(f, order, ("s100", "s110"), "s400")
    c = w["s110"] - w["s100"]
    prev, nxt, fin = w["s100"] - w["s90"], w["s120"] - w["s110"], w["s400"]

    def cos(x, y):
        return (x * y).sum() / np.linalg.norm(x) / np.linalg.norm(y)

    assert st["step_sq"]["s100->s110"] == pytest.approx((c ** 2).sum())
    assert st["cos_prev"] == pytest.approx(cos(c, prev))
    assert st["cos_next"] == pytest.approx(cos(c, nxt))
    assert st["cos_final"] == pytest.approx(cos(c, fin))
    assert st["a_move"] == pytest.approx(
        np.linalg.norm(f["s110"][0] - f["s100"][0]) / np.linalg.norm(f["s100"][0]))


def test_window_directions_are_the_changes_top_singular_vectors():
    rng = np.random.default_rng(1)
    f = {n: factors(rng) for n in ("a", "b")}
    u, s = cd.window_directions(f, ("a", "b"), k=2)
    dense = f["b"][1] @ f["b"][0] - f["a"][1] @ f["a"][0]
    uu, ss, _ = np.linalg.svd(dense)
    assert np.allclose(s, ss[:2])
    assert abs(u[:, 0] @ uu[:, 0]) == pytest.approx(1.0)


def test_lazy_reader_and_logit_lens(tmp_path):
    vals = np.array([[1.0, -2.5], [0.15625, 3.0]], dtype=np.float32)
    h = json.dumps({"x": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
    p = tmp_path / "t.safetensors"
    p.write_bytes(len(h).to_bytes(8, "little") + h
                  + (vals.view(np.uint32) >> 16).astype("<u2").tobytes())
    assert np.array_equal(cd.LazySafetensors(p).get("x"), vals)
    w_u = np.eye(4)
    res = cd.logit_lens(np.array([0.0, 3.0, 0.0, -1.0]), w_u, np.ones(4), None, {"b": 1}, top=1)
    assert res["up"] == [1] and res["down"] == [3] and res["watch_z"]["b"] > 1
