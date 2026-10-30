"""One fixed mixture of two observation-only objectives; no channel labels."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from sahure.data import load_observations, split_observations
from sahure.model import AdaptiveHankelEstimator, ModelConfig
from sahure.math import nll
from sahure_expansion.train import condition_mean, save_checkpoint
from rician_pilot_screen.noise import make_views
from .protocol import ROOT, DATA, CONFIG, write, sha, initialize


class WorkerTimeLimit(RuntimeError):
    pass


def configure_cpu():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise WorkerTimeLimit("time limit reached; restore the last complete epoch")


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def parameters_hash(model):
    digest = hashlib.sha256()
    for value in model.parameters():
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def new_model(seed):
    torch.manual_seed(seed)
    config = ModelConfig(core="em", structure="adaptive", stages=CONFIG["stages"])
    model = AdaptiveHankelEstimator(config).float()
    if sum(p.numel() for p in model.parameters()) != 188:
        raise RuntimeError("expected the unchanged 188-parameter estimator")
    return model, config


def step(model, optimizer, context, target):
    optimizer.zero_grad(set_to_none=True)
    estimate = model(*context.args())
    if not bool(torch.isfinite(estimate).all()):
        raise FloatingPointError("nonfinite channel output")
    loss = nll(target.z, estimate @ target.s + target.b, target.sigma2).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite training likelihood")
    loss.backward()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError("nonfinite gradient; no numerical substitution is permitted")
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("nonfinite parameter after update")
    return float(loss.detach())


def train_epoch(model, optimizer, data, epoch, seed, deadline=float("inf"), progress=None):
    """One forward/backward/update per batch; exactly half use each objective.

    The two losses have different noise levels and must be logged separately.
    Each is a mean over that branch's observations, including its pilot count.
    """
    batch_size = CONFIG["batch_size"]
    if len(data) % (2 * batch_size):
        raise ValueError("an epoch must contain an even number of complete batches")
    check_deadline(deadline)
    augmentation_seed, order_seed = seed + 10000 + epoch, seed + 20000 + epoch
    holdout = split_observations(data, augmentation_seed)
    allpilot = make_views(data, CONFIG["alpha"], seed=augmentation_seed)
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(order_seed))
    branches = (torch.arange(len(data) // batch_size, dtype=torch.int64) + epoch) % 2
    sums = [0.0, 0.0]
    counts = [0, 0]
    batches = [0, 0]
    model.train()
    for batch_number, start in enumerate(range(0, len(data), batch_size)):
        check_deadline(deadline)
        if progress is not None and batch_number % 32 == 0:
            progress(batch_number, len(branches))
        ids = order[start:start + batch_size]
        branch = (batch_number + epoch) % 2
        context, target = holdout if branch == 0 else allpilot
        loss = step(model, optimizer, context.batch(ids), target.batch(ids))
        sums[branch] += loss * len(ids)
        counts[branch] += len(ids)
        batches[branch] += 1
    if batches[0] != batches[1] or counts[0] != counts[1]:
        raise RuntimeError("unbalanced mixed-objective epoch")
    return dict(holdout_train_nll=sums[0] / counts[0], allpilot_train_nll=sums[1] / counts[1],
        holdout_batches=batches[0], allpilot_batches=batches[1],
        holdout_examples=counts[0], allpilot_examples=counts[1],
        augmentation_seed=augmentation_seed, order_seed=order_seed,
        order_sha256=tensor_hash(order), branches_sha256=tensor_hash(branches),
        holdout_input_sha256=tensor_hash(holdout[0].z), holdout_target_sha256=tensor_hash(holdout[1].z),
        allpilot_input_sha256=tensor_hash(allpilot[0].z), allpilot_target_sha256=tensor_hash(allpilot[1].z))


@torch.no_grad()
def validate(model, context, target, conditions, deadline=float("inf")):
    model.eval()
    scores = []
    for start in range(0, len(context), CONFIG["batch_size"]):
        check_deadline(deadline)
        ids = slice(start, start + CONFIG["batch_size"])
        a, b = context.batch(ids), target.batch(ids)
        estimate = model(*a.args())
        score = nll(b.z, estimate @ b.s + b.b, b.sigma2).mean((-2, -1))
        if not bool(torch.isfinite(estimate).all()) or not bool(torch.isfinite(score).all()):
            raise FloatingPointError("nonfinite validation output/likelihood")
        scores.append(score)
    overall, strata = condition_mean(torch.cat(scores).double(), conditions)
    if len(strata) != 126 or not bool(torch.isfinite(strata).all()):
        raise RuntimeError("validation requires all 126 finite strata")
    return float(overall), strata.tolist()


def benchmark():
    """Disposable branch-specific timing; does not freeze or train production models."""
    configure_cpu()
    seed = CONFIG["seeds"][0]
    data = load_observations(DATA / "train_observations.npz").batch(slice(0, CONFIG["batch_size"]))
    views = {"holdout": split_observations(data, seed + 10000),
             "allpilot": make_views(data, CONFIG["alpha"], seed=seed + 10000)}
    timings = {}
    for name, (context, target) in views.items():
        model, _ = new_model(seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
        model.train()
        times = []
        for repeat in range(5):
            tick = time.perf_counter()
            step(model, optimizer, context, target)
            if repeat:
                times.append(time.perf_counter() - tick)
        timings[name] = dict(median_seconds=float(np.median(times)), samples_seconds=times)
    val = load_observations(DATA / "val_observations.npz")
    conditions = np.load(DATA / "val_conditions.npy", allow_pickle=False)
    a, b = split_observations(val, CONFIG["validation_mask_seed"])
    tick = time.perf_counter()
    validate(model, a, b, conditions)
    validation_seconds = time.perf_counter() - tick
    half_batches = CONFIG["train_cases"] // (2 * CONFIG["batch_size"])
    seconds_per_epoch = half_batches * sum(row["median_seconds"] for row in timings.values()) + validation_seconds
    result = dict(parameters=188, batch_size=CONFIG["batch_size"], branch_timings=timings,
        validation_seconds=validation_seconds, estimated_training_seconds=CONFIG["epochs"] * seconds_per_epoch + validation_seconds,
        note="One CPU worker; estimate excludes loading, view generation, hashes and checkpoint writes.")
    write(ROOT / "benchmark.json", result)
    print(json.dumps(result), flush=True)
    return result


def run(seed):
    if seed not in CONFIG["seeds"] or CONFIG["alpha"] != 1.0:
        raise ValueError("unregistered seed or changed fixed re-corruption strength")
    configure_cpu()
    folder = ROOT / "checkpoints" / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        deadline = started + float(CONFIG["max_seconds"])
        completed = 0
        try:
            initialize()
            train = load_observations(DATA / "train_observations.npz")
            val = load_observations(DATA / "val_observations.npz")
            conditions = np.load(DATA / "val_conditions.npy", allow_pickle=False)
            if len(train) != CONFIG["train_cases"] or len(val) != CONFIG["val_cases"]:
                raise RuntimeError("frozen training/validation data size changed")
            a, b = split_observations(val, CONFIG["validation_mask_seed"])
            model, model_config = new_model(seed)
            initial_hash = parameters_hash(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
            record = dict(seed=seed, config=CONFIG, initial_parameters_sha256=initial_hash,
                          train_source_sha256=sha(__file__), torch=torch.__version__)
            manifest = folder / "manifest.json"
            if manifest.exists() and json.loads(manifest.read_text()) != record:
                raise RuntimeError("mixed worker configuration, source, or initialization changed")
            if not manifest.exists():
                write(manifest, record)
            latest, best = folder / "mixed_latest.pt", folder / "mixed_best.pt"
            if latest.exists():
                state = torch.load(latest, map_location="cpu", weights_only=False)
                if state["initial_parameters_sha256"] != initial_hash or state["model_config"] != asdict(model_config):
                    raise RuntimeError("resume checkpoint initialization/configuration mismatch")
                completed = state["epoch"]
                if [r["epoch"] for r in state["history"]] != list(range(1, completed + 1)) or not (0 <= completed <= CONFIG["epochs"]):
                    raise RuntimeError("invalid completed-epoch history")
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
            else:
                write(folder / "status.json", dict(state="initial_validation", seed=seed, pid=os.getpid(), completed_epochs=0, updated_unix=time.time()))
                score, strata = validate(model, a, b, conditions, deadline)
                state = dict(model_config=asdict(model_config), seed=seed, epoch=0, best_epoch=0,
                    initial_parameters_sha256=initial_hash, initial_validation_score=score,
                    best_score=score, history=[], train_seconds=0.0, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), validation_strata=strata)
                save_checkpoint(state, best)
                save_checkpoint(state, latest)
            for epoch in range(completed, CONFIG["epochs"]):
                check_deadline(deadline)
                tick = time.monotonic()
                def progress(batch_number, batches):
                    write(folder / "status.json", dict(state="training", seed=seed, pid=os.getpid(),
                        epoch=epoch + 1, completed_epochs=completed, completed_batches=batch_number,
                        batches=batches, updated_unix=time.time()))
                training = train_epoch(model, optimizer, train, epoch, seed, deadline, progress)
                write(folder / "status.json", dict(state="validating", seed=seed, pid=os.getpid(),
                    epoch=epoch + 1, completed_epochs=completed, updated_unix=time.time()))
                score, strata = validate(model, a, b, conditions, deadline)
                seconds = time.monotonic() - tick
                state.update(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict(), validation_strata=strata)
                state["train_seconds"] += seconds
                row = dict(epoch=epoch + 1, val_nll=score, seconds=seconds, **training)
                state["history"].append(row)
                if score < state["best_score"]:
                    state["best_score"], state["best_epoch"] = score, epoch + 1
                    save_checkpoint(state, best)
                save_checkpoint(state, latest)
                completed = epoch + 1
                write(folder / "history.json", state["history"])
                print(json.dumps(dict(seed=seed, **row, best_score=state["best_score"], best_epoch=state["best_epoch"])), flush=True)
            write(folder / "status.json", dict(state="complete", seed=seed, pid=os.getpid(),
                epochs=CONFIG["epochs"], completed_epochs=completed, best_epoch=state["best_epoch"],
                best_score=state["best_score"], train_seconds=state["train_seconds"], updated_unix=time.time()))
            return True
        except WorkerTimeLimit as exc:
            write(folder / "status.json", dict(state="time_limit", seed=seed, pid=os.getpid(),
                completed_epochs=completed, error=str(exc), partial_epoch_saved=False, updated_unix=time.time()))
            return False
        except Exception as exc:
            write(folder / "status.json", dict(state="failed", seed=seed, pid=os.getpid(),
                completed_epochs=completed, error=repr(exc), traceback=traceback.format_exc(), updated_unix=time.time()))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.benchmark:
        benchmark()
    elif args.seed in CONFIG["seeds"]:
        raise SystemExit(0 if run(args.seed) else 3)
    else:
        parser.error("supply --benchmark or a registered --seed")
