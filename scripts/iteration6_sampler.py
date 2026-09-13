"""Fixed log-variance interpolation; learned weights and the reverse mean are unchanged."""
import numpy as np
import torch

from medhyperdiff.diffusion import (MixedDiffusion, seed_all, clean_from_prediction,
    clean_categorical_logits, categorical_reverse_probs)


def reverse_variance(beta, posterior, mixture, step):
    if not np.isfinite(mixture) or not 0 <= mixture <= 1:
        raise ValueError("variance mixture must be finite and in [0, 1]")
    if step == 1:
        return torch.zeros_like(posterior)
    if mixture == 0:
        return posterior
    if mixture == 1:
        return beta
    return torch.exp(mixture*beta.log()+(1-mixture)*posterior.log())


class VarianceSampler(MixedDiffusion):
    variance_mix = 0.

    def sample(self, labels, seed=0, batch_size=256):
        if not np.isfinite(self.variance_mix) or not 0 <= self.variance_mix <= 1:
            raise ValueError("variance_mix must be finite and in [0, 1]")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        seed_all(seed)
        labels = np.asarray(labels, dtype=np.int64)
        if not np.isin(labels, [0, 1]).all() or len(labels) == 0:
            raise ValueError("Provide a nonempty vector of binary encoded labels")
        self.model.eval()
        cfg = self.config
        all_num, all_cat = [], []
        for start in range(0, len(labels), batch_size):
            y = torch.as_tensor(labels[start:start+batch_size], device=self.device)
            n = len(y)
            numeric = torch.randn(n, len(self.preprocessor.schema.numeric), device=self.device)
            categorical = torch.stack([torch.full((n,), k, device=self.device, dtype=torch.long)
                for k in self.preprocessor.cardinalities], dim=1)
            for t in range(cfg.steps, 0, -1):
                with torch.no_grad():
                    predicted, logits = self.model(numeric, categorical, torch.full((n,), t/cfg.steps, device=self.device), y)
                    clean = clean_from_prediction(predicted, numeric, self.abar[t], cfg.numeric_prediction)
                    logits = clean_categorical_logits(logits, categorical, self.preprocessor.cardinalities)
                clean, logits = self.guide(clean, logits, cfg.guidance*self.schedule[t])
                with torch.no_grad():
                    prev_abar, abar = self.abar[t-1], self.abar[t]
                    c1 = self.beta[t]*prev_abar.sqrt()/(1-abar)
                    c2 = (1-prev_abar)*self.alpha[t].sqrt()/(1-abar)
                    posterior = self.beta[t]*(1-prev_abar)/(1-abar)
                    variance = reverse_variance(self.beta[t], posterior, self.variance_mix, t)
                    numeric = c1*clean+c2*numeric
                    if t > 1:
                        numeric += variance.sqrt()*torch.randn_like(numeric)
                    for j, k in enumerate(self.preprocessor.cardinalities):
                        probs = categorical_reverse_probs(logits[j], categorical[:, j], k, t)
                        categorical[:, j] = torch.multinomial(probs, 1).squeeze(1)
            all_num.append(numeric.cpu().numpy())
            all_cat.append(categorical.cpu().numpy())
        return self.preprocessor.inverse(np.concatenate(all_num), np.concatenate(all_cat))


def distribution_shape(train, validation, synthetic, schema):
    result = {}
    for col in schema.numeric:
        a, b, c = (table[col].dropna().to_numpy(dtype=float) for table in (train, validation, synthetic))
        spread = float(np.std(a, ddof=1)) if len(a) > 1 else 1.
        if not np.isfinite(spread) or spread < 1e-6:
            spread = 1.
        if min(len(a), len(b), len(c)) == 0:
            result[col] = {"status": "insufficient_observed_values"}
            continue
        qa, qb, qc = [np.quantile(x, [.05, .25, .5, .75, .95]) for x in (a, b, c)]
        result[col] = dict(status="evaluated", fit_quantiles=qa.tolist(), validation_quantiles=qb.tolist(),
            synthetic_quantiles=qc.tolist(), quantile_probabilities=[.05, .25, .5, .75, .95], fit_sd=spread,
            q05_error=float(abs(qc[0]-qb[0])/spread), q50_error=float(abs(qc[2]-qb[2])/spread),
            q95_error=float(abs(qc[4]-qb[4])/spread),
            iqr_ratio=float((qc[3]-qc[1])/(qb[3]-qb[1])) if qb[3] > qb[1] else None,
            fit_upper95_rate=float((a > qa[4]).mean()), validation_upper95_rate=float((b > qa[4]).mean()),
            synthetic_upper95_rate=float((c > qa[4]).mean()),
            validation_outside_fit_range=float(((b < a.min()) | (b > a.max())).mean()),
            synthetic_outside_fit_range=float(((c < a.min()) | (c > a.max())).mean()))
    return result
