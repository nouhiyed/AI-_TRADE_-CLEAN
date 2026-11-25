"""Independent outcome ledger for the four observational edge strategies.

Research-only proxy exits: entry at candidate close, stop=1.5 ATR, target=2.25
ATR, 48 subsequent closed 5m bars. These are evaluation labels, NOT trade plans.
Same-bar stop/target ambiguity is resolved stop-first by research_outcomes.
"""
import json
import sqlite3
import threading
import time
from pathlib import Path

from core.research_outcomes import evaluate_forward_path


class EdgeResearchStore:
    def __init__(self, path="data/edge_research.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS edge_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL, symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL, version TEXT NOT NULL,
                    direction TEXT NOT NULL, candle_time TEXT NOT NULL,
                    entry REAL NOT NULL, stop REAL NOT NULL, target REAL NOT NULL,
                    atr REAL NOT NULL, strength REAL NOT NULL, reason TEXT NOT NULL,
                    features_json TEXT NOT NULL, captured_at_ms INTEGER NOT NULL,
                    UNIQUE(market,symbol,strategy,version,candle_time)
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS edge_outcomes (
                    candidate_id INTEGER PRIMARY KEY, status TEXT NOT NULL,
                    bars INTEGER NOT NULL, result_r REAL, mfe_r REAL NOT NULL,
                    mae_r REAL NOT NULL, reason TEXT NOT NULL,
                    labelled_at_ms INTEGER NOT NULL,
                    FOREIGN KEY(candidate_id) REFERENCES edge_candidates(id)
                )
            """)

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=10)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=10000")
        return db

    @staticmethod
    def _records(frame):
        if frame is None:
            return []
        try:
            rows = frame.to_dict("records") if hasattr(frame, "to_dict") else list(frame)
            return [r for r in rows if isinstance(r, dict) and bool(r.get("is_closed", False))]
        except Exception:
            return []

    def record(self, market, symbol, frames, candidates):
        rows = self._records((frames or {}).get("5m"))
        if len(rows) < 15:
            return 0
        last = rows[-1]
        try:
            entry = float(last["close"])
            ranges = []
            for i in range(max(1, len(rows)-14), len(rows)):
                prev = float(rows[i-1]["close"])
                ranges.append(max(float(rows[i]["high"])-float(rows[i]["low"]),
                                  abs(float(rows[i]["high"])-prev),
                                  abs(float(rows[i]["low"])-prev)))
            atr = sum(ranges)/len(ranges)
            candle_time = str(last.get("open_time", last.get("timestamp", "")))
            if not candle_time or atr <= 0:
                return 0
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return 0
        inserted = 0
        with self.lock, self._connect() as db:
            for candidate in (candidates or []):
                direction = str(candidate.get("direction", "")).upper()
                if direction not in ("LONG", "SHORT"):
                    continue
                risk = 1.5 * atr
                stop = entry-risk if direction == "LONG" else entry+risk
                target = entry+2.25*atr if direction == "LONG" else entry-2.25*atr
                cursor = db.execute("""
                    INSERT OR IGNORE INTO edge_candidates
                    (market,symbol,strategy,version,direction,candle_time,entry,stop,
                     target,atr,strength,reason,features_json,captured_at_ms)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (str(market).lower(), str(symbol).upper(),
                      str(candidate.get("strategy","")), str(candidate.get("strategy_version","")),
                      direction,candle_time,entry,stop,target,atr,
                      float(candidate.get("strength",0)),str(candidate.get("reason","")),
                      json.dumps({"atr":atr,"entry":entry},sort_keys=True),
                      int(time.time()*1000)))
                inserted += max(0,cursor.rowcount)
        return inserted

    @staticmethod
    def _stamp(value):
        try:
            n=float(value)
            if n > 10_000_000_000: n/=1000
            return n
        except (TypeError,ValueError):
            return None

    def label(self, market, symbol, bars, horizon=48, fee_rate=0.001, slippage_rate=0.0005):
        clean=[]
        for row in self._records(bars):
            stamp=self._stamp(row.get("open_time",row.get("timestamp")))
            try:
                if stamp is not None: clean.append((stamp,{"high":float(row["high"]),"low":float(row["low"])}))
            except (KeyError,TypeError,ValueError):
                continue
        clean.sort(key=lambda x:x[0])
        with self.lock, self._connect() as db:
            pending=db.execute("""
                SELECT c.id,c.candle_time,c.direction,c.entry,c.stop,c.target
                FROM edge_candidates c LEFT JOIN edge_outcomes o ON o.candidate_id=c.id
                WHERE c.market=? AND c.symbol=? AND o.candidate_id IS NULL
            """,(str(market).lower(),str(symbol).upper())).fetchall()
        count=0
        for cid,candle_time,direction,entry,stop,target in pending:
            stamp=self._stamp(candle_time)
            future=[bar for t,bar in clean if stamp is not None and t>stamp]
            if len(future)<horizon: continue
            outcome=evaluate_forward_path(direction,entry,stop,target,future[:horizon],
                max_bars=horizon,fee_rate=fee_rate,slippage_rate=slippage_rate)
            with self.lock,self._connect() as db:
                cur=db.execute("""INSERT OR IGNORE INTO edge_outcomes
                    (candidate_id,status,bars,result_r,mfe_r,mae_r,reason,labelled_at_ms)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (cid,outcome.status,outcome.bars,outcome.result_r,outcome.mfe_r,
                     outcome.mae_r,outcome.reason,int(time.time()*1000)))
                count+=max(0,cur.rowcount)
        return count

    def summary(self, market=None):
        sql="""SELECT c.strategy,o.status,COUNT(*),AVG(o.result_r),AVG(o.mfe_r),AVG(o.mae_r)
               FROM edge_candidates c JOIN edge_outcomes o ON o.candidate_id=c.id"""
        args=()
        if market:
            sql+=" WHERE c.market=?"; args=(str(market).lower(),)
        sql+=" GROUP BY c.strategy,o.status ORDER BY c.strategy,o.status"
        with self._connect() as db: rows=db.execute(sql,args).fetchall()
        return [{"strategy":r[0],"status":r[1],"count":r[2],"avg_result_r":r[3],
                 "avg_mfe_r":r[4],"avg_mae_r":r[5]} for r in rows]

    def counts(self):
        with self._connect() as db:
            candidates=db.execute("SELECT COUNT(*) FROM edge_candidates").fetchone()[0]
            labelled=db.execute("SELECT COUNT(*) FROM edge_outcomes").fetchone()[0]
        return {"candidates":candidates,"labelled":labelled,
                "status":"INSUFFICIENT_DATA" if labelled<100 else "RESEARCH_ONLY"}
