import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnose_v3 import trade_stats


def test_trade_stats_reports_net_performance():
    result = trade_stats([
        {"net_R": 1.5},
        {"net_R": -1.0},
        {"net_R": 0.5},
    ])
    assert result["trades"] == 3
    assert result["wins"] == 2
    assert result["win_rate_pct"] == 66.67
    assert result["net_R"] == 1.0
    assert result["average_R"] == 0.3333
    assert result["profit_factor"] == 2.0


def test_trade_stats_handles_empty_and_all_winners():
    assert trade_stats([])["trades"] == 0
    result = trade_stats([{"net_R": 1.0}, {"net_R": 2.0}])
    assert result["profit_factor"] == "infinite"
