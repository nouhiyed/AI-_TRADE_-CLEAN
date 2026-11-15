import unittest
import pandas as pd
from core.trailing_stop import calculate_trailing_stop

class TrailingStopTests(unittest.TestCase):
    def frame(self, base=100.0):
        rows=[]
        for i in range(20):
            c=base+i*0.1
            rows.append({"high":c+0.4,"low":c-0.3,"close":c,"ATR":1.0})
        return pd.DataFrame(rows)

    def test_long_stop_only_tightens(self):
        f=self.frame()
        result=calculate_trailing_stop(f,"LONG",99.0,101.0,102.0)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["stop"],99.0)

    def test_short_stop_only_tightens(self):
        f=self.frame()
        result=calculate_trailing_stop(f,"SHORT",103.0,100.0,99.0)
        self.assertIsNotNone(result)
        self.assertLessEqual(result["stop"],103.0)

    def test_invalid_frame_fails_closed(self):
        self.assertIsNone(calculate_trailing_stop(pd.DataFrame([{"close":1}]),"LONG",.9,1,1.1))

    def test_trailing_has_no_time_exit_constant(self):
        from core.scalping_v3 import FuturesScalpingStrategyV3
        plan=FuturesScalpingStrategyV3.plan("futures","LONG",self.frame(), "TEST","TEST",1.5,100)
        self.assertTrue(plan["success"])
        self.assertEqual(plan["max_hold_minutes"],0)

if __name__=="__main__": unittest.main()
