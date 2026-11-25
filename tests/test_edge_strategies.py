import unittest
from core.edge_strategies import EdgeStrategyEngine


class Frame(list):
    attrs = {"timeframe": "5m"}
    def to_dict(self, orient="records"):
        return list(self)


def candles(n=70, start=100.0, step=0.15):
    rows=[]
    price=start
    for i in range(n):
        op=price
        price += step
        rows.append({"open":op,"high":price+0.08,"low":op-0.08,
                     "close":price,"volume":100+i%5,"is_closed":True,
                     "open_time":i})
    return rows


class EdgeStrategyTests(unittest.TestCase):
    def test_rejects_missing_or_unclosed_market_data(self):
        engine=EdgeStrategyEngine()
        self.assertFalse(engine.analyze("futures", {})["success"])
        rows=candles()
        for row in rows:
            row["is_closed"]=False
        result=engine.analyze("futures", {"5m":Frame(rows)})
        self.assertFalse(result["success"])

    def test_spot_never_proposes_short(self):
        result=EdgeStrategyEngine().analyze("spot", {"5m":Frame(candles())})
        self.assertTrue(result["success"])
        self.assertTrue(all(x["direction"]=="LONG" for x in result["candidates"]))

    def test_candidates_are_never_execution_authorized(self):
        result=EdgeStrategyEngine().analyze("futures", {"5m":Frame(candles())})
        self.assertFalse(result["execution_authorized"])
        self.assertTrue(all(x["execution_authorized"] is False for x in result["candidates"]))


if __name__=="__main__":
    unittest.main()
