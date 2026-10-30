"""Matched all-pilot training; channel truth is never loaded here."""

import os
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_variable] = "1"

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

from rician_pilot_screen.noise import make_views
from sahure.data import load_observations, split_observations
from sahure.math import nll
from sahure.model import AdaptiveHankelEstimator, ModelConfig
from sahure_expansion.train import condition_mean, save_checkpoint
from .protocol import ROOT, DATA, CONFIG, initialize, write, sha


class WorkerTimeLimit(RuntimeError):
    """The last complete epoch remains the only resume checkpoint."""


def _configure_cpu():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise WorkerTimeLimit("worker time limit reached; resume from the last complete epoch")


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameters_hash(model):
    """Exact byte concatenation in model.parameters() order for paired auditing."""
    digest = hashlib.sha256()
    for value in model.parameters():
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _new_model(seed):
    torch.manual_seed(seed)
    config = ModelConfig(core="em", structure="adaptive", stages=CONFIG["stages"])
    model = AdaptiveHankelEstimator(config).float()
    if sum(p.numel() for p in model.parameters()) != 188:
        raise RuntimeError("the matched estimator must have exactly 188 parameters")
    return model, config


def _loss(model, context, target):
    estimate = model(*context.args())
    if not bool(torch.isfinite(estimate).all()):
        raise FloatingPointError("nonfinite channel output")
    # The target variance is independent of the variance supplied to the model.
    return nll(target.z, estimate @ target.s + target.b, target.sigma2).mean()


def _step(model, optimizer, context, target):
    optimizer.zero_grad(set_to_none=True)
    loss = _loss(model, context, target)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite training likelihood")
    loss.backward()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError("nonfinite gradient; no substitution is permitted")
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("nonfinite parameter after the Adam update")
    return float(loss.detach())


@torch.no_grad()
def validate(model, context, target, conditions, deadline=float("inf")):
    """Original equal-stratum 21/9 likelihood criterion, including epoch zero."""
    scores = []
    model.eval()
    for start in range(0, len(context), CONFIG["batch_size"]):
        _check_deadline(deadline)
        ids = slice(start, start + CONFIG["batch_size"])
        a, b = context.batch(ids), target.batch(ids)
        estimate = model(*a.args())
        if not bool(torch.isfinite(estimate).all()):
            raise FloatingPointError("nonfinite validation channel output")
        score = nll(b.z, estimate @ b.s + b.b, b.sigma2).mean((-2, -1))
        if not bool(torch.isfinite(score).all()):
            raise FloatingPointError("nonfinite per-example validation likelihood")
        scores.append(score)
    overall, strata = condition_mean(torch.cat(scores).double(), conditions)
    if len(strata) != 126 or not bool(torch.isfinite(strata).all()):
        raise RuntimeError("validation must contain all 126 finite strata")
    return float(overall), strata.tolist()


def benchmark():
    """Five disposable batch updates and one full validation; no checkpoint saved."""
    _configure_cpu()
    seed = CONFIG["seeds"][0]
    data = load_observations(DATA / "train_observations.npz").batch(slice(0, CONFIG["batch_size"]))
    context, target = make_views(data, CONFIG["alpha"], seed=seed + 10000)
    model, _ = _new_model(seed)
    initial_hash = model_hash(model)
    initial_parameters_hash = parameters_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    times = []
    model.train()
    for iteration in range(5):
        tick = time.perf_counter()
        _step(model, optimizer, context, target)
        if iteration:
            times.append(time.perf_counter() - tick)
    val = load_observations(DATA / "val_observations.npz")
    conditions = np.load(DATA / "val_conditions.npy", allow_pickle=False)
    a, b = split_observations(val, CONFIG["validation_mask_seed"])
    tick = time.perf_counter()
    validation_nll, _ = validate(model, a, b, conditions)
    validation_seconds = time.perf_counter() - tick
    batch_seconds = float(np.median(times))
    batches = (CONFIG["train_cases"] + CONFIG["batch_size"] - 1) // CONFIG["batch_size"]
    record = dict(
        parameters=188, batch_size=CONFIG["batch_size"], timed_batches=4,
        batch_seconds=batch_seconds, batch_seconds_individual=times,
        full_validation_seconds=validation_seconds, validation_nll=validation_nll,
        estimated_seconds_per_seed=CONFIG["epochs"] * (batches * batch_seconds + validation_seconds) + validation_seconds,
        initial_model_sha256=initial_hash,
        initial_parameters_sha256=initial_parameters_hash,
        note="Single-worker CPU timing. Parallel workers may slow one another; excludes loading, hashing, and checkpoint writes.",
    )
    write(ROOT / "benchmark.json", record)
    print(json.dumps(record), flush=True)
    return record


def run(seed):
    """Train one fixed seed, safely resuming only a completed epoch."""
    if seed not in CONFIG["seeds"]:
        raise ValueError("unregistered training seed")
    if CONFIG["alpha"] != 1.0:
        raise ValueError("the full comparison freezes alpha at one")
    _configure_cpu()
    folder = ROOT / "checkpoints" / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "run.lock").open("a") as lock:
        # A duplicate cannot overwrite the active worker's status.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed_epochs = 0
        started = time.monotonic()
        deadline = started + float(CONFIG.get("max_seconds", 14400))
        try:
            initialize()
            train = load_observations(DATA / "train_observations.npz")
            val = load_observations(DATA / "val_observations.npz")
            conditions = np.load(DATA / "val_conditions.npy", allow_pickle=False)
            if len(train) != CONFIG["train_cases"] or len(val) != CONFIG["val_cases"]:
                raise RuntimeError("the frozen training/validation data sizes changed")
            a_val, b_val = split_observations(val, CONFIG["validation_mask_seed"])
            model, model_config = _new_model(seed)
            initial_hash = model_hash(model)
            initial_parameters_hash = parameters_hash(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
            repository = Path(__file__).resolve().parent.parent
            paths = [Path(__file__).resolve()] + [repository / p for p in (
                "sahure/model.py", "sahure/math.py", "sahure/data.py", "sahure_expansion/train.py",
                "rician_pilot_screen/noise.py",
            )]
            manifest = dict(seed=seed, config=CONFIG, initial_model_sha256=initial_hash,
                            initial_parameters_sha256=initial_parameters_hash,
                            sources={str(p): sha(p) for p in paths}, torch=torch.__version__)
            manifest_path = folder / "manifest.json"
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError("worker protocol, source, environment, or initialization changed")
            if not manifest_path.exists():
                write(manifest_path, manifest)
            latest, best = folder / "allpilot_latest.pt", folder / "allpilot_best.pt"
            if latest.exists():
                state = torch.load(latest, map_location="cpu", weights_only=False)
                if (state["initial_model_sha256"] != initial_hash
                    or state["initial_parameters_sha256"] != initial_parameters_hash
                    or state["model_config"] != asdict(model_config)):
                    raise RuntimeError("resume checkpoint initialization/configuration mismatch")
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                completed_epochs = state["epoch"]
                if len(state["history"]) != completed_epochs or not (0 <= completed_epochs <= CONFIG["epochs"]):
                    raise RuntimeError("invalid complete-epoch training history")
            else:
                write(folder / "status.json", dict(state="initial_validation", pid=os.getpid(), seed=seed,
                      completed_epochs=0, updated_unix=time.time()))
                initial_score, strata = validate(model, a_val, b_val, conditions, deadline)
                state = dict(model_config=asdict(model_config), epoch=0, best_epoch=0,
                    best_score=initial_score, initial_val_nll=initial_score,
                    initial_validation_score=initial_score, history=[], train_seconds=0.0,
                    seed=seed, initial_model_sha256=initial_hash,
                    initial_parameters_sha256=initial_parameters_hash,
                    model=model.state_dict(), optimizer=optimizer.state_dict(), validation_strata=strata)
                save_checkpoint(state, best)
                save_checkpoint(state, latest)
            for epoch in range(completed_epochs, CONFIG["epochs"]):
                _check_deadline(deadline)
                tick = time.monotonic()
                augmentation_seed = seed + 10000 + epoch
                order_seed = seed + 20000 + epoch
                context, target = make_views(train, CONFIG["alpha"], seed=augmentation_seed)
                order = torch.randperm(len(train), generator=torch.Generator().manual_seed(order_seed))
                order_hash = tensor_hash(order)
                input_hash, target_hash = tensor_hash(context.z), tensor_hash(target.z)
                total = 0.0
                model.train()
                batches = (len(train) + CONFIG["batch_size"] - 1) // CONFIG["batch_size"]
                for start in range(0, len(train), CONFIG["batch_size"]):
                    _check_deadline(deadline)
                    batch_number = start // CONFIG["batch_size"]
                    if batch_number % 32 == 0:
                        write(folder / "status.json", dict(state="training", pid=os.getpid(), seed=seed,
                            epoch=epoch + 1, completed_epochs=completed_epochs, completed_batches=batch_number,
                            batches=batches, updated_unix=time.time()))
                    ids = order[start:start + CONFIG["batch_size"]]
                    total += _step(model, optimizer, context.batch(ids), target.batch(ids)) * len(ids)
                write(folder / "status.json", dict(state="validating", pid=os.getpid(), seed=seed,
                    epoch=epoch + 1, completed_epochs=completed_epochs, updated_unix=time.time()))
                score, strata = validate(model, a_val, b_val, conditions, deadline)
                seconds = time.monotonic() - tick
                improved = score < state["best_score"]
                state.update(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict(), validation_strata=strata)
                state["train_seconds"] += seconds
                history_entry = dict(epoch=epoch + 1, train_nll=total / len(train), val_nll=score,
                    seconds=seconds, augmentation_seed=augmentation_seed, order_seed=order_seed,
                    order_sha256=order_hash, input_view_sha256=input_hash, target_view_sha256=target_hash)
                state["history"].append(history_entry)
                if improved:
                    state["best_score"], state["best_epoch"] = score, epoch + 1
                    save_checkpoint(state, best)
                save_checkpoint(state, latest)
                completed_epochs = epoch + 1
                write(folder / "history.json", state["history"])
                print(json.dumps(dict(seed=seed, **history_entry, best=state["best_score"], best_epoch=state["best_epoch"])), flush=True)
            write(folder / "status.json", dict(state="complete", pid=os.getpid(), seed=seed,
                epochs=CONFIG["epochs"], completed_epochs=completed_epochs, best_epoch=state["best_epoch"],
                best_score=state["best_score"], train_seconds=state["train_seconds"],
                worker_seconds=time.monotonic() - started, updated_unix=time.time()))
            return True
        except WorkerTimeLimit as exc:
            write(folder / "status.json", dict(state="time_limit", pid=os.getpid(), seed=seed,
                completed_epochs=completed_epochs, error=str(exc), worker_seconds=time.monotonic() - started,
                resume="The preceding complete epoch is saved; a partial epoch is discarded.", updated_unix=time.time()))
            print(json.dumps(dict(seed=seed, state="time_limit", completed_epochs=completed_epochs, error=str(exc))), flush=True)
            return False
        except Exception as exc:
            write(folder / "status.json", dict(state="failed", pid=os.getpid(), seed=seed,
                completed_epochs=completed_epochs, error=repr(exc), traceback=traceback.format_exc(), updated_unix=time.time()))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--benchmark", action="store_true")
    arguments = parser.parse_args()
    if arguments.benchmark:
        benchmark()
    elif arguments.seed in CONFIG["seeds"]:
        raise SystemExit(0 if run(arguments.seed) else 3)
    else:
        parser.error("supply --benchmark or a registered --seed")
