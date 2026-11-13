"""Report regressions: incomplete ladders are valid data, not formatting errors."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("run_report", Path(__file__).resolve().parents[1] / "tools/run_report.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)


class ReportTests(unittest.TestCase):
    def render(self, result):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            (root / "output").mkdir()
            (root / "output/summary.json").write_text(json.dumps({"result": result}))
            return rr.report(root)

    def test_unresolved_ladder(self):
        text = self.render({"bracket_pos": [0.4, None], "bracket_neg": [0.4, None],
                            "eta_percent": [None, None, None]})
        self.assertIn("(0.4, ∞]", text)
        self.assertIn("two finite brackets are needed", text)
        self.assertNotIn("η =", text)

    def test_finite_ladder(self):
        text = self.render({"bracket_pos": [0.6, 0.62], "bracket_neg": [0.66, 0.68],
                            "eta_percent": [-4.6875, -6.25, -3.125]})
        self.assertIn("η = -4.69 % [-6.25, -3.12]", text)

    def test_missing_lower_and_delay(self):
        text = self.render({"bracket_pos": [None, 0.3], "delay_bracket_pos": [0.2, None],
                            "delay_eta_percent": [None, None, None]})
        self.assertIn("(?, 0.3]", text)
        self.assertIn("(0.2, ∞]", text)
        self.assertIn("not reached", text)

    def test_zero_and_nonfinite_eta(self):
        self.assertIsNone(rr.eta((0, 0), (0, 0)))
        self.assertIsNone(rr.eta((0.2, None), (0.3, 0.4)))
        self.assertIsNone(rr.eta((0.2, float("inf")), (0.3, 0.4)))

    def test_any_band_fallback_uses_the_recorded_window_verdict(self):
        rows = [{"J": j, "conv": flag, "onset_class": verdict}
                for j, flag, verdict in [(0, "F", "S"), (0.1, "F", "S"), (0.2, "*", "U"), (0.3, "*", "D")]]
        self.assertEqual(rr.bracket(rows), (0.1, 0.3))


if __name__ == "__main__":
    unittest.main()
