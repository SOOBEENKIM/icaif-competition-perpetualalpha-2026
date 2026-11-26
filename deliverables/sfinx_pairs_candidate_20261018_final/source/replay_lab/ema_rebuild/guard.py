"""Small Gaussian VAE guard with train-only preprocessing and separate calibration."""
import numpy as np
import torch
from torch import nn


def inputs(market):
    n = len(market.times)
    x = np.full((n, 21), np.nan)
    logs = np.log(market.close)
    r = np.diff(logs, axis=0)
    for i in range(168, n):
        recent, week = r[i-24:i], r[i-168:i]
        s24 = np.maximum(recent.std(0), 0.0005)
        s168 = np.maximum(week.std(0), 0.0005)
        values = [(logs[i]-logs[i-h]) / (s168*np.sqrt(h)) for h in (6, 24, 72)]
        values += [np.log(s24/s168), np.log(np.maximum(market.volume[i-23:i+1].mean(0), 1e-8)
                   / np.maximum(market.volume[i-167:i+1].mean(0), 1e-8)),
                   np.minimum(recent, 0).__pow__(2).sum(0) / np.maximum((recent**2).sum(0), 1e-12)]
        centered = week-week.mean(0)
        norms = np.sqrt((centered**2).sum(0))
        corr = centered.T @ centered / np.maximum(norms[:, None]*norms[None, :], 1e-12)
        x[i] = np.r_[np.stack(values, axis=1).ravel(), corr[0, 1], corr[0, 2], corr[1, 2]]
    return x


class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = nn.Linear(cfg['input_dim'], cfg['hidden'])
        self.mean = nn.Linear(cfg['hidden'], cfg['latent'])
        self.logvar = nn.Linear(cfg['hidden'], cfg['latent'])
        self.decoder = nn.Linear(cfg['latent'], cfg['hidden'])
        self.output = nn.Linear(cfg['hidden'], cfg['input_dim'])

    def forward(self, x, sample=False):
        h = torch.tanh(self.encoder(x))
        mu, lv = self.mean(h), self.logvar(h).clamp(-8, 8)
        z = mu + torch.exp(0.5*lv)*torch.randn_like(mu) if sample else mu
        return self.output(torch.tanh(self.decoder(z))), mu, lv


def score(x, state):
    scores = np.full(len(x), np.nan)
    valid = np.isfinite(x).all(1)
    z = torch.tensor((x[valid]-state['mean']) / state['scale'], dtype=torch.float32)
    parts = []
    for weights in state['weights']:
        model = VAE(state['config'])
        model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()})
        model.eval()
        with torch.no_grad():
            pred, _, _ = model(z)
            parts.append(((pred-z)**2).mean(1).numpy())
    scores[valid] = np.mean(parts, axis=0)
    return scores


def train(x, times, cfg, train_start, train_end, calibration_start, calibration_end):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train_ids = np.flatnonzero((times >= train_start) & (times < train_end)
                              & (times % 21600 == 0) & np.isfinite(x).all(1))
    calibration_ids = np.flatnonzero((times >= calibration_start) & (times < calibration_end)
                                    & (times % 21600 == 0) & np.isfinite(x).all(1))
    if len(train_ids) < 100 or len(calibration_ids) < 50 or train_end > calibration_start:
        raise ValueError('Insufficient or overlapping VAE partitions')
    mean = x[train_ids].mean(0)
    scale = np.maximum(x[train_ids].std(0), 1e-6)
    z = torch.tensor((x[train_ids]-mean)/scale, dtype=torch.float32)
    state = dict(config=cfg, mean=mean.tolist(), scale=scale.tolist(), weights=[], training=[],
                 train_indices=train_ids.tolist(), calibration_indices=calibration_ids.tolist(),
                 last_train_time=int(times[train_ids[-1]]), last_calibration_time=int(times[calibration_ids[-1]]),
                 active_from=int(calibration_end), supervised_labels=0, performance_seed_selection=False)
    for seed in cfg['seeds']:
        torch.manual_seed(seed)
        model = VAE(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        losses = []
        for epoch in range(cfg['epochs']):
            ids = torch.randperm(len(z))
            total = 0.
            for b in ids.split(cfg['batch_size']):
                pred, mu, lv = model(z[b], sample=True)
                mse = ((pred-z[b])**2).mean()
                kl = -0.5*(1+lv-mu.square()-lv.exp()).mean()
                loss = mse + cfg['kl_weight']*kl
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()*len(b)
            losses.append(total/len(z))
        state['weights'].append({k: v.detach().tolist() for k, v in model.state_dict().items()})
        state['training'].append(dict(seed=seed, epochs=cfg['epochs'], losses=losses))
    errors = score(x[calibration_ids], state)
    state['threshold_enter'] = float(np.quantile(errors, cfg['anomaly_enter_quantile']))
    state['threshold_exit'] = float(np.quantile(errors, cfg['anomaly_exit_quantile']))
    state['calibration_scores'] = errors.tolist()
    return state


def defensive_flags(scores, times, state):
    flags = np.zeros(len(times), dtype=bool)
    defensive = False
    for i, time in enumerate(times):
        if time >= state['active_from'] and time % 21600 == 0:
            if not np.isfinite(scores[i]):
                defensive = True
            elif scores[i] >= state['threshold_enter']:
                defensive = True
            elif scores[i] <= state['threshold_exit']:
                defensive = False
        flags[i] = defensive
    return flags
