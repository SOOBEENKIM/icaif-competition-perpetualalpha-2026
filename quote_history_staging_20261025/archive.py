"""Immutable received-quote feature cache; no network or trading interface."""
import copy
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

from experiments.basis_carry_20261017.probe import asof


class UnflushedBoundary(ValueError):
    """The public capture has not yet stored a receipt past the boundary."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def read_point(paths, point, symbols, cfg):
    """Copy bounded gzip bytes once. Only already-received observations count.

    Open members may end with EOF; malformed records are rejected. A caller
    must supply the capture files covering the point. Missing coverage remains
    missing and is not inferred from trades or external prices.
    """
    if not math.isfinite(point) or point < 0 or point % 3600:
        raise ValueError('Expected an hourly boundary')
    streams = {s: dict(book=[], ctx=[], resets=[]) for s in symbols}
    resets = []
    manifests = []
    total = 0
    for path in sorted(set(map(Path, paths))):
        size = path.stat().st_size
        if size > 64 * 1024**2 or total + size > 256 * 1024**2:
            raise ValueError('Quote archive read exceeds the bounded budget')
        raw = path.read_bytes()
        total += len(raw)
        if len(raw) > 64 * 1024**2 or total > 256 * 1024**2:
            raise ValueError('Quote archive grew beyond the bounded budget')
        record = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                      bytes=len(raw), open_member=False, max_received_ms=-1)
        try:
            with gzip.open(io.BytesIO(raw), 'rt') as stream:
                for line in stream:
                    e = json.loads(line)
                    at = e['received_ms']
                    if not isinstance(at, int) or at < 0:
                        raise ValueError('Invalid receive timestamp')
                    if at < record['max_received_ms']:
                        raise ValueError('Capture receive clock moved backwards')
                    record['max_received_ms'] = at
                    if at > point * 1000:
                        continue
                    channel = e['channel']
                    if channel == 'connection':
                        resets.append(at)
                        continue
                    if channel not in ('l2Book', 'activeAssetCtx'):
                        continue
                    d = e['data']
                    coin = d['coin']
                    if not coin.startswith('pas:') or coin[4:] not in streams:
                        raise ValueError('Unexpected quote venue or universe')
                    field = 'book' if channel == 'l2Book' else 'ctx'
                    age = cfg['book_max_age_ms'] if field == 'book' else cfg['context_max_age_ms']
                    if point * 1000 - at > age:
                        continue
                    # Epoch identity also includes the file's parent capture
                    # directory. Two independent connections never match just
                    # because both counters started at one.
                    epoch = (str(path.parent.resolve()), e['epoch'])
                    streams[coin[4:]][field].append((at, epoch, d if field == 'book' else d['ctx']))
        except EOFError:
            record['open_member'] = True
        manifests.append(record)
    quotes = {}
    for symbol, s in streams.items():
        s['resets'] = sorted(resets)
        for field in ('book', 'ctx'):
            s[field].sort(key=lambda r: r[0])
            # A second, conflicting record at an identical receive time is
            # not resolved by an arbitrary filename ordering.
            for a, b in zip(s[field], s[field][1:]):
                if a[0] == b[0] and a != b:
                    raise ValueError('Ambiguous quote receive timestamp')
            s[field + '_times'] = [r[0] for r in s[field]]
        quotes[symbol] = asof(s, point * 1000, cfg)
    return dict(point=point, quotes=quotes, files=manifests)


class PointCache:
    def __init__(self, directory, symbols, cfg):
        self.directory = Path(directory)
        self.symbols = tuple(symbols)
        self.cfg = copy.deepcopy(cfg)
        self.identity = digest(dict(symbols=self.symbols, cfg=self.cfg, schema=2))

    def get(self, point, paths, *, now=None):
        injected_clock = now is not None
        now = time.time() if now is None else now
        if not math.isfinite(now) or not math.isfinite(point) or point < 0 or point > now or point % 3600:
            raise ValueError('Cannot cache a future or unaligned boundary')
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f'{int(point)}.json'
        with (self.directory / 'writer.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                result = json.loads(path.read_text())
                body = result['body']
                if (result['sha256'] != digest(body) or body['identity'] != self.identity
                        or body['point'] != point or body['created_at'] > now):
                    raise ValueError('Cached history identity, time, or digest mismatch')
                return result
            observation = read_point(paths, point, self.symbols, self.cfg)
            watermark = max((p['max_received_ms'] for p in observation['files']), default=-1)
            if watermark <= point * 1000:
                raise UnflushedBoundary('Archive has not flushed beyond the source boundary')
            created = now if injected_clock else time.time()
            if watermark > created * 1000:
                raise ValueError('Future receive timestamp cannot prove current coverage')
            body = dict(identity=self.identity, created_at=created, **observation)
            result = dict(body=body, sha256=digest(body))
            # Once written, even missing history remains the original evidence.
            # A later file revision cannot rewrite the original decision input.
            temp = path.with_suffix('.tmp')
            with temp.open('w') as out:
                json.dump(result, out, allow_nan=False)
                out.flush()
                os.fsync(out.fileno())
            temp.replace(path)
            return result

    def history(self, source, paths, *, now=None):
        paths = tuple(paths)
        points = []
        for point in (source - 86400, source - 10800, source):
            try:
                points.append(self.get(point, paths, now=now))
            except UnflushedBoundary:
                # Temporarily absent features cannot suppress account/risk
                # processing. They remain missing and are retried next tick.
                points.append(None)
        streams = {s: dict(book=[], ctx=[], resets=[]) for s in self.symbols}
        for epoch, record in enumerate(points):
            if record is None:
                continue
            for symbol, q in record['body']['quotes'].items():
                if q is None:
                    continue
                s = streams[symbol]
                s['resets'].append(min(q['received'].values()))
                for field in ('book', 'ctx'):
                    s[field].append((q['received'][field], epoch, copy.deepcopy(q[field])))
        for s in streams.values():
            for field in ('book', 'ctx'):
                s[field + '_times'] = [r[0] for r in s[field]]
        return streams, [r['sha256'] if r is not None else None for r in points]
