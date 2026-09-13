"""Independent v9 audit of Descending masks and the v8 tolerance contrast.

No production policy functions are imported.  --initialize records the scope
and input hashes before --run reconstructs masks from the frozen row arrays.
Outputs are audit records; frozen scientific evidence is never changed.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEADS = ((220, 220), (660, 220), (220, 660), (660, 660))
INPUTS = (
    "evidence/matched_censoring/complete/CALIBRATION.npz",
    "evidence/matched_censoring/complete/EVALUATION.npz",
    "evidence/matched_censoring/complete/FROZEN.json",
    "evidence/revision_v7_complete_screen/FROZEN.json",
    "evidence/revision_v8_tolerance/RESULTS.json",
    "paper/revision_v7_calibration_table.tex",
    "paper/revision_v7_B_table.tex",
    "code/build_revision_v7_assets.py",
)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(relative):
    return json.loads((ROOT / relative).read_text())


def metrics(a, y, loss):
    weight = float(y[a].sum())
    error = float(loss[a].sum())
    return dict(rows=int(a.sum()), total_rows=int(len(a)),
                c=float(a.mean()), d=weight / float(y.sum()),
                loss=error, weight=weight,
                risk=error / weight if weight > 0 else None,
                mask_sha256=hashlib.sha256(np.packbits(a).tobytes()).hexdigest())


def compare_metrics(actual, recorded):
    for key in ("rows", "c", "d", "loss", "weight", "risk"):
        assert np.isclose(actual[key], recorded[key], rtol=1e-12, atol=1e-9), key


def independent_largest_threshold(score, y, loss, blocks, cap, floor):
    """Replay whole-tie prefix feasibility; no production code reuse."""
    thresholds = np.unique(score)
    feasible = np.ones(len(thresholds), dtype=bool)
    for block in np.unique(blocks):
        ids = np.flatnonzero(blocks == block)
        order = ids[np.argsort(score[ids], kind="stable")]
        counts = np.searchsorted(score[order], thresholds, side="right")
        weights = np.concatenate(([0.0], np.cumsum(y[order])))[counts]
        errors = np.concatenate(([0.0], np.cumsum(loss[order])))[counts]
        feasible &= ((weights > 0) & (counts >= floor * len(ids)) &
                     (errors <= cap * weights + 1e-9))
    good = np.flatnonzero(feasible)
    return None if not len(good) else float(thresholds[good[-1]])


def initialize(out):
    out.mkdir(parents=True, exist_ok=True)
    target = out / "DESCENDING_COVERAGE_PROTOCOL.json"
    if target.exists():
        raise FileExistsError(target)
    save(target, dict(
        utc=now(), scope="Retrospective arithmetic and mask audit prompted by the v8 review; no new experiment or prospective preregistration.",
        checks=[
            "Reconstruct complete-arm Descending masks for all four head cells at the original relative .95 cap.",
            "Compare A/B counts, total rows, exposure sums, mask hashes, exact minima and two-decimal table entries.",
            "Independently replay largest whole-tie thresholds for each distinct exposure head and design.",
            "Compare changed-row membership across exposure heads and identical masks across error heads.",
            "Reconstruct original A-only .95r Descending thresholds and Table 37 metrics.",
            "At epsilon .005, reconstruct all four complete-arm Later Ratio-minus-unchanged-case contrasts and gain/loss ratios from row arrays, then compare frozen v8 results.",
        ], model_fits=0, new_threshold_designs=0, new_draws=0,
        new_external_openings=0, consumed_evaluation_reopened=True,
        code_sha256=sha(__file__),
        inputs_sha256={p: sha(ROOT / p) for p in INPUTS}))
    print("PROTOCOL LOCKED", sha(target))


def run(out):
    protocol_path = out / "DESCENDING_COVERAGE_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    assert protocol["code_sha256"] == sha(__file__)
    for path, expected in protocol["inputs_sha256"].items():
        assert sha(ROOT / path) == expected, path
    frozen = read("evidence/matched_censoring/complete/FROZEN.json")
    screened = read("evidence/revision_v7_complete_screen/FROZEN.json")
    tolerance = read("evidence/revision_v8_tolerance/RESULTS.json")
    with np.load(ROOT / INPUTS[0]) as z:
        data = {k: z[k] for k in z.files}
    y, loss = data["y"], abs(data["y"] - data["f"])
    blocks = np.unique(data["blocks"])
    assert list(blocks) == [0, 1]
    names = {0: "A", 1: "B"}
    rows, screened_rows, masks = [], [], {}
    threshold_replays = []
    for e, w in HEADS:
        plan = next(p for p in frozen["plans"] if
                    p["kind"] == "relative" and p["value"] == .95 and
                    (p["e_trees"], p["w_trees"]) == (e, w))
        candidate = plan["policies"]["weight_descending"]
        score = -data[f"w{w}"]
        accepted = score <= candidate["threshold"]
        masks[e, w] = accepted
        mm = []
        for block, expected in zip(blocks, candidate["calibration"]):
            belongs = data["blocks"] == block
            m = metrics(accepted[belongs], y[belongs], loss[belongs])
            compare_metrics(m, expected)
            m["window"] = names[int(block)]
            mm.append(m)
        min_c, min_d = min(mm, key=lambda m: m["c"]), min(mm, key=lambda m: m["d"])
        rows.append(dict(e_trees=e, w_trees=w, threshold=candidate["threshold"],
                         cap=plan["cap"], windows=mm,
                         min_case_percent=100*min_c["c"], min_case_window=min_c["window"],
                         min_exposure_percent=100*min_d["d"], min_exposure_window=min_d["window"],
                         printed_case=f'{100*min_c["c"]:.2f}',
                         printed_exposure=f'{100*min_d["d"]:.2f}'))
        if e == 220:
            t = independent_largest_threshold(score, y, loss, data["blocks"], plan["cap"], .35)
            assert t == candidate["threshold"]
            threshold_replays.append(dict(design="A/B", w_trees=w, threshold=t))
        splan = next(p for p in screened["plans"] if
                     (p["e_trees"], p["w_trees"]) == (e, w))
        sc = splan["policies"]["weight_descending"]
        sa = score <= sc["threshold"]
        smm = []
        for block in blocks:
            belongs = data["blocks"] == block
            m = metrics(sa[belongs], y[belongs], loss[belongs])
            compare_metrics(m, sc[names[int(block)]])
            m["window"] = names[int(block)]
            smm.append(m)
        screened_rows.append(dict(e_trees=e, w_trees=w, threshold=sc["threshold"],
                                  design_cap=screened["design_cap"],
                                  report_cap=screened["cap"], windows=smm,
                                  A_case_percent=100*smm[0]["c"],
                                  printed_A_case=f'{100*smm[0]["c"]:.2f}'))
        if e == 220:
            belongs = data["blocks"] == 0
            t = independent_largest_threshold(score[belongs], y[belongs], loss[belongs],
                                              data["blocks"][belongs], screened["design_cap"], .35)
            assert t == sc["threshold"]
            threshold_replays.append(dict(design="A-only", w_trees=w, threshold=t))
    assert np.array_equal(masks[220, 220], masks[660, 220])
    assert np.array_equal(masks[220, 660], masks[660, 660])
    membership = []
    low, high = masks[220, 220], masks[220, 660]
    for block in blocks:
        belongs = data["blocks"] == block
        both = low & high & belongs
        only_low = low & ~high & belongs
        only_high = high & ~low & belongs
        membership.append(dict(window=names[int(block)],
            identical_masks=bool(np.array_equal(low[belongs], high[belongs])),
            both_rows=int(both.sum()), only_w220_rows=int(only_low.sum()),
            only_w660_rows=int(only_high.sum()), changed_rows=int((only_low|only_high).sum()),
            only_w220_exposure=float(y[only_low].sum()),
            only_w660_exposure=float(y[only_high].sum()),
            net_row_difference_w660_minus_w220=int(only_high.sum()-only_low.sum()),
            net_exposure_difference_w660_minus_w220=float(y[only_high].sum()-y[only_low].sum())))
    assert len({r["printed_case"] for r in rows}) == 1
    assert rows[0]["min_case_percent"] != rows[2]["min_case_percent"]
    # Verify the displayed source actually contains each reconstructed pair.
    table = (ROOT / "paper/revision_v7_calibration_table.tex").read_text()
    for r in rows:
        assert (f'{r["e_trees"]}/{r["w_trees"]} & Descending & '
                f'{r["printed_case"]} & {r["printed_exposure"]}') in table

    with np.load(ROOT / INPUTS[1]) as z:
        evaluation = {k: z[k] for k in z.files}
    ey, eloss = evaluation["y"], abs(evaluation["y"] - evaluation["f"])
    contrasts = []
    for setting in tolerance["settings"]:
        if setting["arm"] != "complete":
            continue
        e, w, cap = setting["e_trees"], setting["w_trees"], setting["cap"]
        band = next(b for b in setting["bands"] if b["epsilon"] == .005)
        assert band["selected"] == {"c": "row", "d": "demand"}
        excess = evaluation[f"e{e}"] - cap * evaluation[f"w{w}"]
        den = evaluation[f"w{w}"] / setting["T"]
        ratio_score = np.divide(excess, den, out=np.where(excess > 0, np.inf,
                                np.where(excess < 0, -np.inf, 0.0)), where=den > 0)
        cm = metrics(excess <= setting["candidates"]["row"]["threshold"], ey, eloss)
        dm = metrics(ratio_score <= setting["candidates"]["demand"]["threshold"], ey, eloss)
        compare_metrics(cm, band["evaluation"]["case_selected"])
        compare_metrics(dm, band["evaluation"]["exposure_selected"])
        dc, dd = dm["c"] - cm["c"], dm["d"] - cm["d"]
        assert np.isclose(dc, band["evaluation"]["contrast_vs_case"]["c"], rtol=1e-12)
        assert np.isclose(dd, band["evaluation"]["contrast_vs_case"]["d"], rtol=1e-12)
        assert dc < 0 and dd > 0
        contrasts.append(dict(e_trees=e, w_trees=w, epsilon_pp=.5,
                              case_choice="Excess", exposure_choice="Ratio",
                              case_metrics=cm, exposure_metrics=dm,
                              delta_case_pp=100*dc, delta_exposure_pp=100*dd,
                              exchange_ratio=dd/-dc))
    assert len(contrasts) == 4
    ranges = {k: [min(c[k] for c in contrasts), max(c[k] for c in contrasts)]
              for k in ("delta_case_pp", "delta_exposure_pp", "exchange_ratio")}
    report = dict(status="PASS", utc=now(), protocol_sha256=sha(protocol_path),
                  code_sha256=sha(__file__), input_hashes_verified=True,
                  original_AB_records=rows, original_A_only_records=screened_rows,
                  whole_tie_threshold_replays=threshold_replays,
                  exposure_head_membership_comparison=membership,
                  error_head_invariance="For each fixed exposure head, Descending scores, thresholds and masks are exactly identical across error heads; the score is -predicted exposure and does not use the error head.",
                  result="The two distinct exposure heads produce unequal exact minimum case coverage that rounds to 48.67% in both cases; their masks and accepted exposure sums differ. The stricter A-only design uses different thresholds and a smaller design cap, yielding 35.05%/35.09%. No copy or transfer discrepancy was found.",
                  eq7_complete_half_pp_contrasts=contrasts,
                  eq7_complete_half_pp_ranges=ranges,
                  limitations="Deterministic audit of consumed data and frozen decisions; no additional generalization, population-risk, tuning, or new-seed evidence.")
    target = out / "DESCENDING_COVERAGE_CHECK.json"
    save(target, report)
    print(json.dumps(dict(status="PASS", original_case_percent=[r["min_case_percent"] for r in rows],
                          membership=membership, eq7_ranges=ranges), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction_outputs/revision_v9_coverage")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    initialize(args.output) if args.initialize else run(args.output)
