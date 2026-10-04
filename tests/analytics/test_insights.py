"""Static checks on the hypothesis queries; execution is covered against the live service."""
import unittest

from forza_ai.analytics import insights


class HypothesisQueryTests(unittest.TestCase):
    def test_six_hypotheses_in_order(self):
        self.assertEqual([title[:2] for title, _, _ in insights.HYPOTHESES], ["H1", "H2", "H3", "H4", "H5", "H6"])

    def test_every_query_is_session_filterable_and_read_only(self):
        for title, hint, sql in insights.HYPOTHESES:
            with self.subTest(title=title):
                self.assertIn("%(session)s", sql)
                self.assertTrue(hint)
                upper = sql.upper()
                for verb in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER"):
                    self.assertNotIn(verb, upper)

    def test_ai_only_queries_exclude_human_laps(self):
        for query in (insights.H1_UNDERSTEER, insights.H2_SPEED, insights.H4_LAG):
            self.assertIn("source IN ('session', 'runtime')", query)
        self.assertIn("source = 'recorder'", insights.H5_COVERAGE)


if __name__ == "__main__":
    unittest.main()
