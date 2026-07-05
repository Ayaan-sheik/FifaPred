"""Tests for knockout_payload() in build_predictions.py (module-level function).

Uses the real bracket + dataset (same fixture scope as test_bracket_state.py)
to verify structural invariants.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Real-data fixture (module-scoped for speed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ko_payload_data():
    """Load real bracket/dataset, fit model, apply form, build ko payload."""
    from fifa_pred.build_predictions import knockout_payload
    from fifa_pred.bracket_state import build_tournament_state
    from fifa_pred.data_loader import load_bracket, load_results, merge_live_results
    from fifa_pred.dixon_coles import DixonColes
    from fifa_pred.tournament_form import apply_form, compute_form
    from fifa_pred.wc_dataset import completed_results_frame, load_dataset

    bracket = load_bracket()
    ds = load_dataset()
    live = completed_results_frame(ds)
    df = merge_live_results(load_results(), live)
    model = DixonColes.fit(df)
    eff_base = model.effective_params(bracket)
    form = compute_form(ds, bracket, model)
    eff = apply_form(eff_base, form)
    state = build_tournament_state(ds, bracket)
    rows = knockout_payload(state, eff, model)
    return rows, state, eff


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKnockoutPayloadStructure:
    def test_exactly_31_entries(self, ko_payload_data):
        """tournament.knockout must have exactly 31 rows (73–102, 104)."""
        rows, _, _ = ko_payload_data
        assert len(rows) == 31

    def test_match_ids_73_to_102_and_104(self, ko_payload_data):
        """match_ids must be exactly {73..102} ∪ {104}."""
        rows, _, _ = ko_payload_data
        expected = set(range(73, 103)) | {104}
        actual = {r["match_id"] for r in rows}
        assert actual == expected

    def test_no_match_103(self, ko_payload_data):
        """Match 103 (third-place) must not appear."""
        rows, _, _ = ko_payload_data
        assert all(r["match_id"] != 103 for r in rows)

    def test_schema_uniform(self, ko_payload_data):
        """Every row must have home_placeholder, away_placeholder, and p_home_win keys."""
        rows, _, _ = ko_payload_data
        for r in rows:
            assert "home_placeholder" in r, f"match {r['match_id']} missing home_placeholder"
            assert "away_placeholder" in r, f"match {r['match_id']} missing away_placeholder"
            assert "p_home_win" in r, f"match {r['match_id']} missing p_home_win"

    def test_completed_rows_have_p_home_win_none(self, ko_payload_data):
        """Completed rows must have p_home_win = None."""
        rows, _, _ = ko_payload_data
        for r in rows:
            if r["status"].upper() == "COMPLETED":
                assert r["p_home_win"] is None, (
                    f"match {r['match_id']} is COMPLETED but p_home_win={r['p_home_win']}"
                )

    def test_sorted_by_stage_then_bracket_position(self, ko_payload_data):
        """Rows must be grouped by stage (R32 -> R16 -> QF -> SF -> Final) and,
        within each stage, ordered by bracket_position — NOT by match_id."""
        rows, _, _ = ko_payload_data
        from fifa_pred.bracket_state import _STAGE_RANK
        keys = [(_STAGE_RANK.get(r["stage"], 0), r["bracket_position"]) for r in rows]
        assert keys == sorted(keys)

    def test_r32_rows_follow_true_bracket_draw_not_match_id(self, ko_payload_data):
        """R32 display order must match the real bracket tree (e.g. match 88
        is not necessarily the last slot in the draw)."""
        rows, _, _ = ko_payload_data
        r32_ids = [r["match_id"] for r in rows if 73 <= r["match_id"] <= 88]
        assert r32_ids == [
            75, 78, 73, 76, 84, 83, 82, 81,
            74, 77, 79, 80, 87, 86, 85, 88,
        ]
        assert r32_ids != sorted(r32_ids)

    def test_bracket_position_present_on_every_row(self, ko_payload_data):
        rows, _, _ = ko_payload_data
        for r in rows:
            assert isinstance(r["bracket_position"], int)


class TestPendingMatches:
    def test_match_89_has_p_home_win_in_range(self, ko_payload_data):
        """Match 89 (CAN vs PAR, pending) must have p_home_win ∈ (0, 1)."""
        rows, _, _ = ko_payload_data
        m89 = next((r for r in rows if r["match_id"] == 89), None)
        assert m89 is not None
        pw = m89["p_home_win"]
        assert pw is not None and 0.0 < pw < 1.0, f"p_home_win={pw} must be in (0,1)"

    def test_stronger_side_has_p_home_win_over_half(self, ko_payload_data):
        """For R16 synthetic matches with resolved participants, stronger side wins > 50%."""
        rows, _, eff = ko_payload_data
        from fifa_pred.bracket_state import _STAGE_OF
        for r in rows:
            if (
                r["status"] == "Pending"
                and r["home_id"] is not None
                and r["away_id"] is not None
                and r["p_home_win"] is not None
            ):
                h_overall = eff.get(r["home_id"], {}).get("overall", 0.0)
                a_overall = eff.get(r["away_id"], {}).get("overall", 0.0)
                if h_overall > a_overall:
                    assert r["p_home_win"] > 0.5, (
                        f"match {r['match_id']}: stronger home (overall={h_overall:.3f}) "
                        f"has p_home_win={r['p_home_win']:.3f} < 0.5"
                    )
                elif a_overall > h_overall:
                    assert r["p_home_win"] < 0.5


class TestPlaceholderRows:
    def test_qf_rows_have_placeholders(self, ko_payload_data):
        """QF matches (97–100) must have placeholder text for at least one side."""
        rows, _, _ = ko_payload_data
        qf_rows = [r for r in rows if 97 <= r["match_id"] <= 100]
        for r in qf_rows:
            has_placeholder = (
                r.get("home_placeholder") is not None
                or r.get("away_placeholder") is not None
            )
            assert has_placeholder, (
                f"QF match {r['match_id']} has no placeholder text"
            )

    def test_placeholder_format(self, ko_payload_data):
        """Placeholder strings follow 'Winner of <stage> #<mid>' format."""
        rows, _, _ = ko_payload_data
        for r in rows:
            for key in ("home_placeholder", "away_placeholder"):
                v = r.get(key)
                if v is not None:
                    assert v.startswith("Winner of "), (
                        f"match {r['match_id']} {key}={v!r} does not start with 'Winner of '"
                    )
                    assert "#" in v, f"match {r['match_id']} {key}={v!r} missing '#'"


class TestResolvedR16Rows:
    def test_r16_synthetic_rows_have_real_ids(self, ko_payload_data):
        """R16 matches 90–96 (synthetic, all R32 complete) must have real team ids."""
        rows, state, _ = ko_payload_data
        r16_synthetic = [r for r in rows if 90 <= r["match_id"] <= 96]
        for r in r16_synthetic:
            assert r["home_id"] is not None, (
                f"R16 match {r['match_id']} home_id should be resolved (all R32 done)"
            )
            assert r["away_id"] is not None, (
                f"R16 match {r['match_id']} away_id should be resolved (all R32 done)"
            )
