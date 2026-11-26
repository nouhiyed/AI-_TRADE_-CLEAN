"""Append-only research capture for every completed V3 analysis.

This module is observational only: it never changes signals, plans, risk, or
execution. SQLite writes are serialized and duplicate 5m candles are ignored.
"""
import json
import math
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from core.research_outcomes import evaluate_forward_path
from core.research_features import extract_research_features, classify_market_regime


def _clean(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _clean(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return str(value)


def _latest_row(frame):
    if frame is None or len(frame) == 0:
        return {}
    try:
        row = frame.iloc[-1]
        return {str(key): _clean(value) for key, value in row.to_dict().items()}
    except Exception:
        return {}


class ResearchCandidateStore:
    """Durable candidate ledger, independent from the trading journal."""

    def __init__(self, path="data/research_candidates.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    candle_open_time TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    direction TEXT,
                    setup TEXT,
                    strategy_version TEXT,
                    reason TEXT,
                    entry REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    features_json TEXT NOT NULL,
                    signal_json TEXT NOT NULL,
                    captured_at_ms INTEGER NOT NULL,
                    UNIQUE(market, symbol, timeframe, candle_open_time, profile)
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS idx_candidates_market_time "
                       "ON candidates(market, captured_at_ms)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_candidates_decision "
                       "ON candidates(market, decision, setup)")
            db.execute("""
                CREATE TABLE IF NOT EXISTS stress_outcomes (
                    candidate_id INTEGER NOT NULL,
                    scenario TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_r REAL,
                    bars INTEGER NOT NULL,
                    fee_rate REAL NOT NULL,
                    slippage_rate REAL NOT NULL,
                    recorded_at_ms INTEGER NOT NULL,
                    PRIMARY KEY(candidate_id, scenario),
                    FOREIGN KEY(candidate_id) REFERENCES candidates(id)
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS outcomes (
                    candidate_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    bars INTEGER NOT NULL,
                    exit_price REAL,
                    result_r REAL,
                    mfe_r REAL NOT NULL,
                    mae_r REAL NOT NULL,
                    reason TEXT NOT NULL,
                    labelled_at_ms INTEGER NOT NULL,
                    FOREIGN KEY(candidate_id) REFERENCES candidates(id)
                )
            """)

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=10)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=10000")
        return db

    def record(self, market, symbol, profile, frames, signal):
        market = str(market).lower()
        symbol = str(symbol).upper()
        frames = frames if isinstance(frames, dict) else {}
        signal = signal if isinstance(signal, dict) else {}
        five = frames.get("5m")
        latest = _latest_row(five)
        candle_time = latest.get("open_time", latest.get("open_time_ms",
                       latest.get("timestamp", latest.get("time"))))
        if candle_time is None:
            # Missing source candle identity is not silently deduplicated by
            # wall-clock time; preserve the record and make the limitation clear.
            candle_time = "unknown:" + str(int(time.time() * 1000))
        plan = signal.get("trade_plan")
        plan = plan if isinstance(plan, dict) else {}
        diagnostics = signal.get("diagnostics")
        payload = {
            "diagnostics": diagnostics or plan.get("diagnostics"),
            "plan_reason": plan.get("reason"),
            "setup": plan.get("setup"),
            "market_context": signal.get("market_context"),
            "signal_reason": signal.get("reason"),
            "source": signal.get("signal_source"),
        }
        features = {tf: _latest_row(frame) for tf, frame in frames.items()}
        derived = extract_research_features(frames)
        features["research"] = derived
        payload["research_regime"] = classify_market_regime(derived)
        decision = "ACCEPTED" if plan.get("success") is True else "REJECTED"
        values = (
            market, symbol, "5m", str(candle_time), str(profile or "standard"),
            decision, str(signal.get("direction") or ""),
            str(plan.get("setup") or signal.get("setup") or ""),
            str(signal.get("strategy_version") or ""),
            str(plan.get("reason") or signal.get("reason") or ""),
            _clean(plan.get("entry")), _clean(plan.get("stop_loss")),
            _clean(plan.get("take_profit")),
            json.dumps(_clean(features), separators=(",", ":"), sort_keys=True),
            json.dumps(_clean(payload), separators=(",", ":"), sort_keys=True),
            int(time.time() * 1000),
        )
        with self._lock, self._connect() as db:
            cursor = db.execute("""
                INSERT OR IGNORE INTO candidates (
                    market,symbol,timeframe,candle_open_time,profile,decision,
                    direction,setup,strategy_version,reason,entry,stop_loss,
                    take_profit,features_json,signal_json,captured_at_ms
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, values)
            candidate = db.execute("""
                SELECT id FROM candidates
                WHERE market=? AND symbol=? AND timeframe=? AND candle_open_time=? AND profile=?
            """, (market, symbol, "5m", str(candle_time), str(profile or "standard"))).fetchone()
            return {"inserted": cursor.rowcount == 1,
                    "candidate_id": candidate[0] if candidate else None,
                    "decision": decision, "symbol": symbol,
                    "candle_open_time": str(candle_time)}

    def record_outcome(self, candidate_id, outcome):
        """Attach one immutable forward label to an existing candidate."""
        required = ("status", "bars", "mfe_r", "mae_r", "reason")
        if any(not hasattr(outcome, key) for key in required):
            raise ValueError("outcome object is incomplete")
        values = (str(outcome.status), int(outcome.bars),
                  _clean(outcome.exit_price), _clean(outcome.result_r),
                  float(outcome.mfe_r), float(outcome.mae_r),
                  str(outcome.reason), int(time.time() * 1000), int(candidate_id))
        with self._lock, self._connect() as db:
            cursor = db.execute("""
                INSERT OR IGNORE INTO outcomes
                (status,bars,exit_price,result_r,mfe_r,mae_r,reason,labelled_at_ms,candidate_id)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, values)
            return cursor.rowcount == 1

    def record_stress_outcome(self, candidate_id, scenario, outcome,
                              fee_rate, slippage_rate):
        with self._lock, self._connect() as db:
            cursor = db.execute("""
                INSERT OR IGNORE INTO stress_outcomes
                (candidate_id,scenario,status,result_r,bars,fee_rate,
                 slippage_rate,recorded_at_ms)
                VALUES (?,?,?,?,?,?,?,?)
            """, (int(candidate_id), str(scenario), str(outcome.status),
                  _clean(outcome.result_r), int(outcome.bars),
                  float(fee_rate), float(slippage_rate), int(time.time() * 1000)))
            return cursor.rowcount == 1

    def stress_summary(self, market=None):
        query = """
            SELECT s.scenario,s.status,COUNT(*),AVG(s.result_r),AVG(s.fee_rate),
                   AVG(s.slippage_rate)
            FROM stress_outcomes s JOIN candidates c ON c.id=s.candidate_id
        """
        params = ()
        if market:
            query += " WHERE c.market=?"
            params = (str(market).lower(),)
        query += " GROUP BY s.scenario,s.status ORDER BY s.scenario,s.status"
        with self._connect() as db:
            rows = db.execute(query, params).fetchall()
        return [{"scenario": r[0], "status": r[1], "count": r[2],
                 "avg_result_r": r[3], "fee_rate": r[4],
                 "slippage_rate": r[5]} for r in rows]

    def outcome_summary(self, market=None):
        query = """SELECT o.status, COUNT(*), AVG(o.result_r), AVG(o.mfe_r),
                          AVG(o.mae_r)
                   FROM outcomes o JOIN candidates c ON c.id=o.candidate_id"""
        params = ()
        if market:
            query += " WHERE c.market=?"
            params = (str(market).lower(),)
        query += " GROUP BY o.status ORDER BY o.status"
        with self._connect() as db:
            rows = db.execute(query, params).fetchall()
        return [{"status": r[0], "count": r[1], "avg_result_r": r[2],
                 "avg_mfe_r": r[3], "avg_mae_r": r[4]} for r in rows]

    @staticmethod
    def _parse_candle_time(value):
        text = str(value)
        try:
            number = float(text)
            # Binance epoch milliseconds or seconds.
            if number > 10_000_000_000:
                number /= 1000.0
            return datetime.fromtimestamp(number, tz=timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError):
            pass
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
        except (TypeError, ValueError):
            return None

    def label_pending_from_bars(self, market, symbol, bars, horizon=48,
                                fee_rate=0.001, slippage_rate=0.0005):
        """Label valid accepted plans from strictly later, confirmed closed 5m bars.

        Rejected candidates and plans without valid levels remain unlabelled;
        no synthetic entry/stop/target is fabricated for them.
        """
        market, symbol = str(market).lower(), str(symbol).upper()
        rows = []
        for bar in bars or []:
            if isinstance(bar, dict):
                row = bar
            else:
                try:
                    row = bar.to_dict()
                except Exception:
                    continue
            if not bool(row.get("is_closed", True)):
                continue
            rows.append(row)
        rows.sort(key=lambda row: self._parse_candle_time(
            row.get("open_time", row.get("open_time_ms", row.get("timestamp", row.get("time"))))
        ) or datetime.min.replace(tzinfo=timezone.utc))

        with self._lock, self._connect() as db:
            pending = db.execute("""
                SELECT c.id,c.candle_open_time,c.direction,c.entry,c.stop_loss,c.take_profit
                FROM candidates c LEFT JOIN outcomes o ON o.candidate_id=c.id
                WHERE c.market=? AND c.symbol=? AND c.decision='ACCEPTED'
                  AND o.candidate_id IS NULL
                  AND c.entry IS NOT NULL AND c.stop_loss IS NOT NULL
                  AND c.take_profit IS NOT NULL
                ORDER BY c.captured_at_ms ASC
            """, (market, symbol)).fetchall()

        labelled = 0
        for candidate_id, candle_time, direction, entry, stop, target in pending:
            candidate_dt = self._parse_candle_time(candle_time)
            if candidate_dt is None:
                continue
            future = []
            for row in rows:
                stamp = self._parse_candle_time(
                    row.get("open_time", row.get("open_time_ms", row.get("timestamp", row.get("time"))))
                )
                if stamp is not None and stamp > candidate_dt:
                future.append({"high": row["high"], "low": row["low"], "close": row["close"]})
            if not future:
                continue
            outcome = evaluate_forward_path(
                direction, entry, stop, target, future[:int(horizon)],
                max_bars=int(horizon), fee_rate=fee_rate,
                slippage_rate=slippage_rate,
            )
            # An incomplete path is never persisted as a final outcome.
            if outcome.status == "INCOMPLETE":
                continue
            if self.record_outcome(candidate_id, outcome):
                labelled += 1
            scenarios = (
                ("fee_10bps_slip_5bps", 0.0010, 0.0005),
                ("fee_15bps_slip_10bps", 0.0015, 0.0010),
                ("fee_20bps_slip_20bps", 0.0020, 0.0020),
            )
            for name, fee, slip in scenarios:
                stressed = evaluate_forward_path(
                    direction, entry, stop, target, future[:int(horizon)],
                    max_bars=int(horizon), fee_rate=fee,
                    slippage_rate=slip,
                )
                if stressed.status == "INCOMPLETE":
                    continue
                self.record_stress_outcome(
                    candidate_id, name, stressed, fee, slip
                )
        return labelled

    def labelled_dataset(self, market=None, setup=None):
        """Return only completed, accepted and forward-labelled candidates.

        Numeric features are flattened with timeframe prefixes to form a
        reproducible modelling table; labels are never inferred for rejects.
        """
        query = """
            SELECT c.market,c.symbol,c.candle_open_time,c.setup,c.direction,
                   c.features_json,o.result_r,o.status,o.bars
            FROM candidates c JOIN outcomes o ON o.candidate_id=c.id
            WHERE c.decision='ACCEPTED' AND o.result_r IS NOT NULL
        """
        clauses, params = [], []
        if market:
            clauses.append("c.market=?")
            params.append(str(market).lower())
        if setup:
            clauses.append("c.setup=?")
            params.append(str(setup))
        if clauses:
            query += " AND " + " AND ".join(clauses)
        query += " ORDER BY c.candle_open_time ASC,c.symbol ASC"
        with self._connect() as db:
            rows = db.execute(query, tuple(params)).fetchall()
        dataset = []
        for mkt, symbol, candle_time, setup_name, direction, raw, result_r, status, bars in rows:
            stamp = self._parse_candle_time(candle_time)
            if stamp is None:
                continue
            features = {}
            try:
                frames = json.loads(raw or "{}")
            except (TypeError, ValueError):
                frames = {}
            for timeframe, values in frames.items():
                if not isinstance(values, dict):
                    continue
                for key, value in values.items():
                    number = _clean(value)
                    if isinstance(number, (int, float)) and not isinstance(number, bool):
                        features[str(timeframe) + "__" + str(key)] = float(number)
            dataset.append({
                "timestamp": stamp.timestamp(),
                "market": mkt, "symbol": symbol, "setup": setup_name,
                "direction": direction, "features": features,
                "result_r": float(result_r), "outcome": status, "bars": int(bars),
            })
        return dataset

    def summary(self, market=None):
        query = "SELECT decision, COUNT(*) FROM candidates"
        params = ()
        if market:
            query += " WHERE market=?"
            params = (str(market).lower(),)
        query += " GROUP BY decision"
        with self._connect() as db:
            counts = {row[0]: row[1] for row in db.execute(query, params)}
            total = sum(counts.values())
            return {"total": total, "accepted": counts.get("ACCEPTED", 0),
                    "rejected": counts.get("REJECTED", 0)}
