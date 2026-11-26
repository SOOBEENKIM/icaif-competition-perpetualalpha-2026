"""Past-only paired targets, rolling models, and recurrent residual forecasts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import xgboost as xgb

torch.set_num_threads(1)


def design(m, cfg):
    n = len(m.times)
    pairs = [[m.symbols.index(s) for s in p] for p in cfg['pairs']]
    logs = np.log(m.close)
    rets = pd.DataFrame(logs).diff()
    means = rets.rolling(168).mean().to_numpy()
    stdev = rets.rolling(168).std(ddof=0).to_numpy()
    fields, sequence, betas, sigmas, fresh = [], [], [], [], []
    names = []
    for k, (a, b) in enumerate(pairs):
        ra, rb = rets[a], rets[b]
        cov = ra.rolling(168).cov(rb, ddof=0).to_numpy()
        beta = np.clip(cov / np.maximum(stdev[:, b] ** 2, 1e-10), .25, 4.)
        beta_long = np.clip(ra.rolling(720).cov(rb, ddof=0).to_numpy() /
                            np.maximum(rb.rolling(720).var(ddof=0).to_numpy(), 1e-10), .25, 4.)
        vol = np.sqrt(np.maximum(stdev[:, a] ** 2 + beta ** 2 * stdev[:, b] ** 2 - 2 * beta * cov, 1e-8)) / (1 + beta)
        values = []
        columns = []
        def add(name, value):
            columns.append(name); values.append(value)
        for h in (1, 6, 24, 72, 168, 336):
            diff = pd.DataFrame(logs).diff(h).to_numpy()
            relative = (diff[:, a] - beta * diff[:, b]) / (1 + beta)
            add('spread_return_' + str(h), relative / (vol * np.sqrt(h)))
            add('common_return_' + str(h), (diff[:, a] + diff[:, b]) /
                np.maximum((stdev[:, a] + stdev[:, b]) * np.sqrt(h), 1e-8))
        for h in (24, 168, 720):
            mean = pd.DataFrame(logs).rolling(h).mean().to_numpy()
            residual = (logs[:, a] - mean[:, a] - beta * (logs[:, b] - mean[:, b])) / (1 + beta)
            add('spread_deviation_' + str(h), residual / (vol * np.sqrt(h)))
        add('beta', beta)
        add('beta_change', beta / beta_long - 1)
        add('correlation_168', cov / np.maximum(stdev[:, a] * stdev[:, b], 1e-8))
        add('correlation_24', ra.rolling(24).corr(rb).to_numpy())
        # Dynamics of the current fixed hedge, not differences in a changing
        # beta times price level (which would manufacture spread movements).
        diff24 = pd.DataFrame(logs).diff(24).to_numpy()
        add('vol_ratio', (ra.rolling(24).std(ddof=0).to_numpy() + rb.rolling(24).std(ddof=0).to_numpy()) /
            np.maximum(stdev[:, a] + stdev[:, b], 1e-8))
        for h in (24, 168):
            rates = pd.DataFrame(m.funding).shift(1).rolling(h).mean().to_numpy()
            add('funding_' + str(h), (rates[:, a] - beta * rates[:, b]) / (1 + beta) / vol)
        for j in range(len(pairs)):
            add('pair_' + str(j), np.full(n, float(k == j)))
        if names:
            assert names == columns
        names = columns
        fields.append(np.stack(values, axis=-1))
        # Five sequences describe each leg, their hedged return, common return,
        # and differential funding. Each historical row uses then-known beta.
        rra, rrb = ra.to_numpy(), rb.to_numpy()
        sequence.append(np.stack([rra / np.maximum(stdev[:, a], 1e-4), rrb / np.maximum(stdev[:, b], 1e-4),
            (rra - beta * rrb) / (1 + beta) / vol, (rra + rrb) / np.maximum(stdev[:, a] + stdev[:, b], 1e-4),
            (m.funding[:, a] - beta * m.funding[:, b]) / (1 + beta) / vol], axis=-1))
        betas.append(beta); sigmas.append(vol)
        fresh.append((m.volume[:, a] > 0) & (m.volume[:, b] > 0) & (m.trades[:, a] > 0) & (m.trades[:, b] > 0))
    return dict(x=np.clip(np.stack(fields, axis=1), -8, 8), seq=np.nan_to_num(np.clip(np.stack(sequence, axis=1), -8, 8)),
        beta=np.stack(betas, axis=1), vol=np.stack(sigmas, axis=1), fresh=np.stack(fresh, axis=1), pairs=pairs, names=names)


def labels(m, d, horizon):
    n, p = d['beta'].shape
    y = np.full((n, p), np.nan)
    maturity = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    for i in range(n - horizon - 1):
        entry, end = i + 1, i + horizon + 1
        maturity[i] = m.times[end]
        for k, (a, b) in enumerate(d['pairs']):
            if not (d['fresh'][entry, k] and d['fresh'][end, k]):
                continue
            beta = d['beta'][i, k]
            movement = (m.close[end, a] / m.close[entry, a] - 1 - beta * (m.close[end, b] / m.close[entry, b] - 1)) / (1 + beta)
            funding = np.sum(m.close[entry + 1:end + 1, a] / m.close[entry, a] * m.funding[entry + 1:end + 1, a]
                - beta * m.close[entry + 1:end + 1, b] / m.close[entry, b] * m.funding[entry + 1:end + 1, b]) / (1 + beta)
            y[i, k] = (movement - funding) / (d['vol'][i, k] * np.sqrt(horizon))
    return y, maturity


class Recurrent(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.gru = nn.GRU(5, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + features + 1, 24), nn.Tanh(), nn.Linear(24, 2))

    def forward(self, seq, x, linear):
        _, hidden = self.gru(seq)
        return self.head(torch.cat([hidden[-1], x, linear[:, None]], dim=1))


def sequences(seq, indexes, pairs, width):
    return seq[indexes[:, None] - np.arange(width - 1, -1, -1)[None, :], pairs[:, None]].astype('float32')


def fit_predict(m, cfg, out):
    out = Path(out); out.mkdir(exist_ok=True)
    d = design(m, cfg)
    n, count, _ = d['x'].shape
    forecasts = {f'{model}_{h}': np.zeros((n, count)) for h in cfg['horizons'] for model in cfg['models']}
    fit_stamps = np.full((n, count), -1, dtype=np.int64)
    records = []
    dates = [int(np.datetime64(t, 's').astype('int64')) for t in cfg['fit_dates']]
    for horizon in cfg['horizons']:
        target, maturity = labels(m, d, horizon)
        for fold, end in enumerate(dates):
            next_end = dates[fold + 1] if fold + 1 < len(dates) else int(m.times[-1]) + 1
            valid = np.isfinite(d['x']).all(-1) & np.isfinite(target) & d['fresh']
            time_mask = (m.times >= end - cfg['fit_trailing_hours'] * 3600) & (maturity < end) & (m.times % (cfg['fit_sampling_hours'] * 3600) == 0)
            ti, pj = np.where(valid & time_mask[:, None])
            vi, vj = np.where(np.isfinite(d['x']).all(-1) & ((m.times >= end) & (m.times < next_end))[:, None])
            assert len(ti) >= 300 and max(maturity[ti]) < end <= min(m.times[vi])
            x = d['x'][ti, pj]; y = np.clip(target[ti, pj], -4, 4)
            mean = x.mean(0); scale = np.maximum(x.std(0), 1e-6)
            z = (x - mean) / scale; zv = (d['x'][vi, vj] - mean) / scale
            # An empirical rank transform is not learned from validation/test
            # returns. Standardization statistics are training-only here.
            zz = np.c_[np.ones(len(z)), z]
            penalty = np.eye(zz.shape[1]) * cfg['ridge_penalty']; penalty[0, 0] = 0
            coef = np.linalg.solve(zz.T @ zz / len(z) + penalty, zz.T @ y / len(z))
            linear = zz @ coef; linear_v = np.c_[np.ones(len(zv)), zv] @ coef
            params = {k: v for k, v in cfg['tree'].items() if k != 'rounds'}
            params.update(objective='reg:squarederror', tree_method='hist', verbosity=0)
            tree = xgb.train(params, xgb.DMatrix(x, label=y), num_boost_round=cfg['tree']['rounds'])
            tree_v = tree.predict(xgb.DMatrix(d['x'][vi, vj]))
            # Recurrent residual correction follows the linear/nonlinear
            # decomposition, with all labels matured before the fit date.
            torch.manual_seed(cfg['gru']['seed'] + horizon + fold)
            rng = np.random.default_rng(cfg['gru']['seed'] + horizon + fold)
            net = Recurrent(z.shape[-1], cfg['gru']['hidden'])
            optimizer = torch.optim.AdamW(net.parameters(), lr=cfg['gru']['learning_rate'], weight_decay=cfg['gru']['weight_decay'])
            tensor_seq = torch.from_numpy(sequences(d['seq'], ti, pj, cfg['sequence_hours']))
            tensor_x = torch.from_numpy(z.astype('float32')); tensor_l = torch.from_numpy(linear.astype('float32'))
            tensor_y = torch.from_numpy(np.stack([y, y - linear], axis=1).astype('float32'))
            net.train()
            for _ in range(cfg['gru']['epochs']):
                permutation = rng.permutation(len(ti))
                for offset in range(0, len(ti), cfg['gru']['batch']):
                    ix = permutation[offset:offset + cfg['gru']['batch']]
                    optimizer.zero_grad()
                    pred = net(tensor_seq[ix], tensor_x[ix], tensor_l[ix])
                    loss = torch.nn.functional.huber_loss(pred, tensor_y[ix], delta=1.)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.); optimizer.step()
            net.eval(); residual = []
            with torch.no_grad():
                for offset in range(0, len(vi), 512):
                    ii, jj = vi[offset:offset + 512], vj[offset:offset + 512]
                    residual.extend(net(torch.from_numpy(sequences(d['seq'], ii, jj, cfg['sequence_hours'])),
                        torch.from_numpy(zv[offset:offset + 512].astype('float32')),
                        torch.from_numpy(linear_v[offset:offset + 512].astype('float32'))).numpy())
            residual = np.array(residual)
            # A shared recurrent encoder has direct-return and residual heads.
            # Residual predictions alone are never treated as total returns.
            values = dict(ridge=linear_v, tree=tree_v, gru=residual[:, 0], hybrid=linear_v + residual[:, 1],
                          ensemble=(linear_v + tree_v + linear_v + residual[:, 1]) / 3)
            relative_24 = d['x'][vi, vj, d['names'].index('spread_return_24')]
            relative_168 = d['x'][vi, vj, d['names'].index('spread_return_168')]
            values['momentum'] = .25 * relative_24
            values['reversion'] = -.25 * relative_168
            for name, prediction in values.items():
                forecasts[f'{name}_{horizon}'][vi, vj] = np.clip(prediction, -2, 2) * d['vol'][vi, vj] * np.sqrt(horizon)
            fit_stamps[vi, vj] = end
            stem = f'{end}_{horizon}'
            tree.save_model(out / f'{stem}.ubj')
            torch.save(net.state_dict(), out / f'{stem}.pt')
            record = dict(fit_time=end, horizon=horizon, samples=len(ti), train_first=int(m.times[ti].min()),
                last_label_time=int(maturity[ti].max()), prediction_first=int(m.times[vi].min()), prediction_last=int(m.times[vi].max()),
                mean=mean.tolist(), scale=scale.tolist(), beta=coef.tolist(), features=d['names'],
                gru_role='Shared encoder, direct-return head plus residual head; hybrid = linear + residual head')
            (out / f'{stem}.json').write_text(json.dumps(record, indent=2))
            records.append(record)
            print('FIT', stem, 'rows', len(ti), 'predict', len(vi), flush=True)
    np.savez_compressed(out / 'predictions.npz', **forecasts, beta=d['beta'], vol=d['vol'], fresh=d['fresh'], fit_stamps=fit_stamps, times=m.times)
    (out / 'fit_manifest.json').write_text(json.dumps(records, indent=2))
    return forecasts, d, fit_stamps
