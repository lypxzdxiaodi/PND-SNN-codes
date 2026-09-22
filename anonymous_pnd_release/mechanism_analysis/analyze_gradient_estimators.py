"""Monte-Carlo variance diagnostic for hard-spike gradient estimators."""

import argparse
import numpy as np
import torch

from .common import prepare_output, save_json, summary


def probability(evidence, mapping):
    if mapping == "exponential":
        return -torch.expm1(-torch.relu(evidence))
    return torch.sigmoid(evidence)


def estimate(value, mapping, estimator, repeats):
    gradients = []
    for _ in range(repeats):
        evidence = torch.tensor(value, requires_grad=True)
        p = probability(evidence, mapping)
        if estimator == "bernoulli_probability_st":
            sample = torch.bernoulli(p)
            spike = sample - p.detach() + p
        else:
            safe = p.clamp(1e-7, 1 - 1e-7)
            logits = torch.stack([torch.log1p(-safe), torch.log(safe)])
            sample = torch.nn.functional.gumbel_softmax(logits, hard=True)[1]
            if estimator == "gumbel_probability_st":
                sample = sample.detach()
                spike = sample - p.detach() + p
            else:
                spike = sample
        loss = (spike - 0.7).square()
        gradients.append(torch.autograd.grad(loss, evidence)[0].item())
    return np.asarray(gradients)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=4096)
    parser.add_argument("--grid", type=float, nargs="+", default=[-2, -1, -0.25, 0.25, 1, 2, 4])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    output = prepare_output(args.output)
    raw, metrics = {}, {}
    for mapping in ("exponential", "sigmoid"):
        for estimator in ("hard_gumbel_st", "gumbel_probability_st", "bernoulli_probability_st"):
            key = f"{mapping}__{estimator}"
            values = [estimate(point, mapping, estimator, args.repeats) for point in args.grid]
            raw[key] = np.stack(values)
            metrics[key] = {str(point): summary(samples) for point, samples in zip(args.grid, values)}
    np.savez_compressed(output / "raw_data.npz", **raw)
    save_json(output / "config.json", vars(args))
    save_json(output / "metrics.json", {
        "gradient_distributions": metrics,
        "scope": "local stochastic-gradient bias/variance diagnostic; not an accuracy claim",
    })


if __name__ == "__main__":
    main()
