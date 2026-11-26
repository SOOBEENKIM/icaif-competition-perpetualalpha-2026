"""Evaluation reservations survive renamed studies and copied datasets.

This is an application control, not a filesystem security boundary. Archived
research scripts can still reproduce historical results. Such results are not
eligible for a new independent-test claim. Failed evaluations consume their
reservation too: changing a model or run directory cannot reset a holdout.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / 'replay_lab/output/evaluation_control/registry.sqlite3'


class EvaluationRejected(ValueError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class Registry:
    """Conservative latest-period protocol for this competition project.

    All registered exposure shares one research scope. Another venue, model,
    asset subset, file name or train/test split does not reset that scope.
    Unknown provenance is rejected; importing an audit is required first.
    """

    def __init__(self, path=DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS audit (
                    digest TEXT PRIMARY KEY, recorded_at REAL NOT NULL,
                    payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS exposure (
                    id TEXT PRIMARY KEY, start INTEGER NOT NULL,
                    end INTEGER NOT NULL, reason TEXT NOT NULL,
                    evidence TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS trial (
                    id TEXT PRIMARY KEY, frozen_at REAL NOT NULL,
                    payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS access (
                    trial_id TEXT PRIMARY KEY, opened_at REAL NOT NULL,
                    start INTEGER NOT NULL, end INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS result (
                    trial_id TEXT PRIMARY KEY, recorded_at REAL NOT NULL,
                    payload TEXT NOT NULL);
            ''')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute('PRAGMA foreign_keys=ON')
        return db

    def import_audit(self, audit):
        rows = audit['exposures']
        if not rows or not audit.get('metadata_inventory_complete'):
            raise EvaluationRejected('Complete the project metadata audit first')
        payload = canonical(audit)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT OR IGNORE INTO audit VALUES (?,?,?)',
                       (digest, time.time(), payload))
            for row in rows:
                start, end = int(row['start']), int(row['end'])
                if start > end:
                    raise EvaluationRejected('Invalid exposure interval')
                evidence = canonical(row['evidence'])
                ident = hashlib.sha256(canonical(row).encode()).hexdigest()
                db.execute('INSERT OR IGNORE INTO exposure VALUES (?,?,?,?,?)',
                           (ident, start, end, row['reason'], evidence))
        return digest

    @staticmethod
    def _eligibility(db, start, end):
        if not db.execute('SELECT 1 FROM audit LIMIT 1').fetchone():
            raise EvaluationRejected('No project exposure audit: provenance unknown')
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            raise EvaluationRejected('A fixed, increasing test interval is required')
        latest = db.execute('SELECT MAX(end) FROM exposure').fetchone()[0]
        # No earlier-history escape hatch: this is a latest-period protocol.
        if latest is None or start <= latest:
            raise EvaluationRejected(f'Previously exposed history through {latest}; test starts {start}')
        hit = db.execute('SELECT trial_id FROM access WHERE start <= ? AND end >= ? LIMIT 1',
                         (end, start)).fetchone()
        if hit:
            raise EvaluationRejected('Test outcomes already opened in trial ' + hit[0])

    def preflight(self, start, end):
        with self.connect() as db:
            self._eligibility(db, start, end)
        return {'eligible_interval': True, 'evaluation_executed': False}

    @staticmethod
    def _verify_artifacts(plan):
        artifacts = plan.get('artifacts', {})
        if not artifacts:
            raise EvaluationRejected('Freeze candidate, controls, settings and evaluator artifacts')
        for path, digest in artifacts.items():
            p = Path(path)
            if not p.is_absolute() or not p.is_file() or sha(p) != digest:
                raise EvaluationRejected('Frozen artifact changed or missing: ' + path)

    @staticmethod
    def _validate_plan(plan):
        # Intervals are [start, end) for fitting/selection. Test includes its
        # closing equity snapshot at test_end. No label may cross a boundary.
        required = ('train_start', 'train_end', 'validation_start', 'validation_end',
                    'test_start', 'test_end', 'max_train_label_maturity',
                    'preprocessing_fit_last_time', 'selection_last_time',
                    'model_knowledge_last_time')
        if any(type(plan.get(k)) is not int for k in required):
            raise EvaluationRejected('Integer UTC data/label/selection provenance is required')
        if not (plan['train_start'] < plan['train_end'] <= plan['validation_start']
                < plan['validation_end'] <= plan['test_start'] < plan['test_end']):
            raise EvaluationRejected('Train, validation and test are not chronological')
        if not (plan['max_train_label_maturity'] < plan['train_end']
                and plan['preprocessing_fit_last_time'] < plan['train_end']):
            raise EvaluationRejected('Training labels or preprocessing cross training boundary')
        if plan['selection_last_time'] >= plan['test_start']:
            raise EvaluationRejected('Selection has seen test-period observations')
        if plan['model_knowledge_last_time'] >= plan['test_start']:
            raise EvaluationRejected('A model/checkpoint has seen the test period')
        if not plan.get('success_rule') or not plan.get('comparators'):
            raise EvaluationRejected('Freeze success rule and comparator roster before test')
        if plan.get('evaluation_kind') != 'independent_latest_period':
            raise EvaluationRejected('Historical reproduction is a separate research use')

    def freeze(self, plan, *, now=None):
        now = time.time() if now is None else now
        self._validate_plan(plan)
        self._verify_artifacts(plan)
        if plan['test_start'] <= now:
            raise EvaluationRejected('This prospective protocol must freeze before its test starts')
        payload = canonical(plan)
        ident = hashlib.sha256(payload.encode()).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._eligibility(db, plan['test_start'], plan['test_end'])
            # Also prevent registering many different nominees concurrently for
            # one future holdout and then presenting only the best.
            for old_id, old_json in db.execute('SELECT id,payload FROM trial'):
                old = json.loads(old_json)
                if old['test_start'] <= plan['test_end'] and old['test_end'] >= plan['test_start']:
                    if old_id == ident:
                        return ident
                    raise EvaluationRejected('Overlapping independent trial already frozen: ' + old_id)
            db.execute('INSERT INTO trial VALUES (?,?,?)', (ident, now, payload))
        return ident

    def consume(self, trial_id, *, now=None):
        """Reserve BEFORE loading any test observation, not after its result."""
        now = time.time() if now is None else now
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            found = db.execute('SELECT payload FROM trial WHERE id=?', (trial_id,)).fetchone()
            if found is None:
                raise EvaluationRejected('Unknown/unfrozen trial')
            plan = json.loads(found[0])
            self._eligibility(db, plan['test_start'], plan['test_end'])
            if now < plan['test_end']:
                raise EvaluationRejected('Fixed test period has not finished; do not inspect partial results')
            self._verify_artifacts(plan)
            db.execute('INSERT INTO access VALUES (?,?,?,?)',
                       (trial_id, now, plan['test_start'], plan['test_end']))
        return plan

    def evaluate(self, trial_id, load_and_evaluate, *, now=None):
        # If loading, inference or accounting raises, access remains consumed.
        plan = self.consume(trial_id, now=now)
        result = load_and_evaluate(plan)
        self._verify_artifacts(plan)
        with self.connect() as db:
            db.execute('INSERT INTO result VALUES (?,?,?)',
                       (trial_id, time.time(), canonical(result)))
        return result
