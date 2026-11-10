"""Training and test-set evaluation."""

import os
import pickle

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm


def train(model, config, train_loader, valid_loader, folder):
    """Train the model and keep the weights with the lowest validation loss.

    `config` is the "train" block of the configuration. Writes `model.pth` (best weights)
    and `model_last.pth` (final weights) to `folder`.
    """
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    milestones = [int(0.75 * config["epochs"]), int(0.9 * config["epochs"])]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    use_cuda = str(model.device).startswith("cuda")
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    best_path = os.path.join(folder, "model.pth")
    best_loss = float("inf")

    for epoch in range(config["epochs"]):
        model.train()
        running = 0.0
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as bar:
            for n, batch in enumerate(bar, start=1):
                optimizer.zero_grad()
                with torch.autocast("cuda", enabled=use_cuda):
                    loss = model(batch)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                running += loss.item()
                bar.set_postfix(avg_epoch_loss=running / n, epoch=epoch, refresh=False)
                if n >= config["itr_per_epoch"]:
                    break
        scheduler.step()

        if (epoch + 1) % config["valid_epoch_interval"] == 0:
            valid_loss = validate(model, valid_loader, config["max_valid_batches"], use_cuda, epoch)
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), best_path)
                print(f"\nBest validation loss {valid_loss:.6f} at epoch {epoch + 1}. Saved {best_path}")

    torch.save(model.state_dict(), os.path.join(folder, "model_last.pth"))
    model.load_state_dict(torch.load(best_path, map_location=model.device))


def validate(model, valid_loader, max_batches, use_cuda, epoch):
    """Mean validation loss over the first `max_batches` batches, averaged over all diffusion steps."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad(), tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as bar:
        for n, batch in enumerate(bar, start=1):
            if n > max_batches:
                break
            with torch.autocast("cuda", enabled=use_cuda):
                total += model(batch, is_train=False).item()
            count += 1
            bar.set_postfix(valid_avg_epoch_loss=total / n, epoch=epoch, refresh=False)
    return total / count


def evaluate(model, test_loader, n_samples, scaler, mean_scaler, folder, max_batches=None):
    """Sample the test galaxies and save `generated_outputs_nsample<N>.pk`.

    The file holds [samples (N, n_samples, L, 1), truth (N, L, 1), hidden-point mask,
    observed mask, timepoints, per-wavelength std, per-wavelength mean], all in standardised units.
    """
    model.eval()
    samples_all, truth_all, hidden_all, observed_all, time_all = [], [], [], [], []
    sq_error, abs_error, n_points = 0.0, 0.0, 0.0
    with torch.no_grad(), tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as bar:
        for n, batch in enumerate(bar, start=1):
            if max_batches is not None and n > max_batches:
                break
            samples, truth, hidden, timepoints = model.evaluate(batch, n_samples)
            samples = samples.permute(0, 1, 3, 2)
            truth, hidden = truth.permute(0, 2, 1), hidden.permute(0, 2, 1)
            median = samples.median(dim=1).values

            samples_all.append(samples)
            truth_all.append(truth)
            hidden_all.append(hidden)
            observed_all.append(torch.ones_like(truth))
            time_all.append(timepoints)

            error = (median - truth) * hidden
            sq_error += (error ** 2).sum().item()
            abs_error += error.abs().sum().item()
            n_points += hidden.sum().item()
            bar.set_postfix(rmse_normalized=np.sqrt(sq_error / n_points), batch_no=n, refresh=True)

    output = [
        torch.cat(samples_all).cpu(), torch.cat(truth_all).cpu(), torch.cat(hidden_all).cpu(),
        torch.cat(observed_all).cpu(), torch.cat(time_all).cpu(), scaler.cpu().numpy(), mean_scaler.cpu().numpy(),
    ]
    with open(os.path.join(folder, f"generated_outputs_nsample{n_samples}.pk"), "wb") as f:
        pickle.dump(output, f)
    print("RMSE (standardised units):", np.sqrt(sq_error / n_points))
    print("MAE  (standardised units):", abs_error / n_points)
