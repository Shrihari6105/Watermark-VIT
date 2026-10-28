"""SQLite registry: exams, pseudonymous rosters, issued copies, sealed scripts, audit log.

The watermark payload only carries a roster *index*; the mapping back to a
registration number lives here and should be access-controlled.
"""
import datetime as dt
import json
import sqlite3
from pathlib import Path

from .config import EXAM_BITS, ROSTER_BITS, TARDOS_LEN
from .keys import keyed_rng

SCHEMA = """
CREATE TABLE IF NOT EXISTS exams (
    code INTEGER PRIMARY KEY, name TEXT, created TEXT,
    n_students INTEGER, tardos_len INTEGER);
CREATE TABLE IF NOT EXISTS students (
    exam_code INTEGER, roster_idx INTEGER, reg_no TEXT, issued_sha256 TEXT,
    PRIMARY KEY (exam_code, roster_idx));
CREATE TABLE IF NOT EXISTS scripts (
    script_id TEXT PRIMARY KEY, exam_code INTEGER, roster_idx INTEGER,
    sealed_sha256 TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS audit (ts TEXT, action TEXT, detail TEXT);
"""


def _now():
    return dt.datetime.now().isoformat(timespec="seconds")


class Registry:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def log(self, action, **detail):
        self.db.execute("INSERT INTO audit VALUES (?,?,?)", (_now(), action, json.dumps(detail)))
        self.db.commit()

    def create_exam(self, master, name, reg_nos, tardos_len=TARDOS_LEN):
        reg_nos = [r.strip() for r in reg_nos if r.strip()]
        if not reg_nos:
            raise ValueError("roster is empty")
        if len(reg_nos) > (1 << ROSTER_BITS):
            raise ValueError(f"roster larger than {1 << ROSTER_BITS}")
        used = {r["code"] for r in self.db.execute("SELECT code FROM exams")}
        free = [c for c in range(1 << EXAM_BITS) if c not in used]
        if not free:
            raise ValueError("all exam codes in use")
        code = free[0]
        # Pseudonymous index: a keyed shuffle, so indices reveal nothing about order
        perm = keyed_rng(master, f"roster/{code}/{name}").permutation(len(reg_nos))
        self.db.execute("INSERT INTO exams VALUES (?,?,?,?,?)",
                        (code, name, _now(), len(reg_nos), tardos_len))
        self.db.executemany("INSERT INTO students VALUES (?,?,?,NULL)",
                            [(code, int(perm[i]), reg) for i, reg in enumerate(reg_nos)])
        self.db.commit()
        self.log("create_exam", code=code, name=name, n=len(reg_nos))
        return code

    def exam(self, code):
        row = self.db.execute("SELECT * FROM exams WHERE code=?", (code,)).fetchone()
        return dict(row) if row else None

    def exams(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM exams ORDER BY code")]

    def students(self, code):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM students WHERE exam_code=? ORDER BY roster_idx", (code,))]

    def student(self, code, idx):
        row = self.db.execute("SELECT * FROM students WHERE exam_code=? AND roster_idx=?",
                              (code, idx)).fetchone()
        return dict(row) if row else None

    def mark_issued(self, code, idx, sha):
        self.db.execute("UPDATE students SET issued_sha256=? WHERE exam_code=? AND roster_idx=?",
                        (sha, code, idx))
        self.db.commit()

    def add_script(self, script_id, code, idx, sha):
        self.db.execute("INSERT OR REPLACE INTO scripts VALUES (?,?,?,?,?)",
                        (script_id, code, idx, sha, _now()))
        self.db.commit()

    def script(self, script_id):
        row = self.db.execute("SELECT * FROM scripts WHERE script_id=?", (script_id,)).fetchone()
        return dict(row) if row else None

    def audit(self, limit=50):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM audit ORDER BY rowid DESC LIMIT ?", (limit,))]
