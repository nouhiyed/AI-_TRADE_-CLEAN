import tempfile
import unittest
from pathlib import Path
from core.edge_research_store import EdgeResearchStore


class Frame(list):
    def to_dict(self, orient="records"):
        return list(self)


def bars(n=70):
    rows=[]
    for i in range(n):
        p=100+i*.1
        rows.append({"open":p,"high":p+.3,"low":p-.3,"close":p+.1,
                     "volume":100,"is_closed":True,"open_time":i*300000})
    return rows


class EdgeResearchStoreTests(unittest.TestCase):
    def test_records_unique_strategy_candidates_and_labels_only_full_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=EdgeResearchStore(Path(tmp)/"edge.db")
            frames={"5m":Frame(bars())}
            candidates=[
                {"strategy":"PULLBACK","strategy_version":"edge-v1","direction":"LONG",
                 "strength":.6,"reason":"test"},
                {"strategy":"BREAKOUT","strategy_version":"edge-v1","direction":"LONG",
                 "strength":.7,"reason":"test"},
            ]
            self.assertEqual(store.record("futures","BTCUSDT",frames,candidates),2)
            self.assertEqual(store.record("futures","BTCUSDT",frames,candidates),0)
            self.assertEqual(store.counts()["candidates"],2)
            # A later full horizon is required; the candidate candle is excluded.
            future=[dict(x,open_time=(70+i)*300000) for i,x in enumerate(bars(48))]
            self.assertEqual(store.label("futures","BTCUSDT",future,horizon=48),2)
            self.assertEqual(store.counts()["labelled"],2)
            self.assertEqual({x["strategy"] for x in store.summary()},
                             {"PULLBACK","BREAKOUT"})


if __name__=="__main__":
    unittest.main()
