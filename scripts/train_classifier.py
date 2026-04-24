from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import (  # noqa: E402
    GeocellHead,
    build_smoothing_targets,
    save_classifier,
    soft_cross_entropy,
)
from src.embedder import pick_device  # noqa: E402
from src.geocells import cell_centroids_array, load_cells  # noqa: E402
from src.metrics import haversine_km  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP head on StreetCLIP embeddings")
    parser.add_argument("--emb", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--cells", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/classifier.pt")
    parser.add_argument("--curves", type=str, default="results/curves.png")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tau-km", type=float, default=75.0, help="Haversine smoothing temp in km; use 0 for one-hot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    emb = np.load(args.emb)
    meta = pd.read_parquet(args.meta)
    if len(emb) != len(meta):
        raise RuntimeError(f"emb rows {len(emb)} != meta rows {len(meta)}")
    cells, cell_of_index = load_cells(args.cells)
    if len(cell_of_index) != len(meta):
        raise RuntimeError("cells vs meta length mismatch")

    num_cells = len(cells)
    embed_dim = emb.shape[1]
    print(f"[train] N={len(emb)}, D={embed_dim}, cells={num_cells}, tau_km={args.tau_km}")

    centroids = cell_centroids_array(cells)
    cell_ids = np.ascontiguousarray(meta["geocell_id"].to_numpy(dtype=np.int64))
    lats = np.ascontiguousarray(meta["lat"].to_numpy(dtype=np.float64))
    lons = np.ascontiguousarray(meta["lon"].to_numpy(dtype=np.float64))
    is_val = (meta["split"].to_numpy() == "val")
    train_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val)
    print(f"[train] train={len(train_idx)}, val={len(val_idx)}")

    if args.tau_km > 0:
        smoothing = build_smoothing_targets(lats, lons, cell_ids, centroids)
    else:
        smoothing = None

    device = pick_device()
    model = GeocellHead(embed_dim, args.hidden_dim, num_cells, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    emb_t = torch.from_numpy(emb)
    cell_t = torch.from_numpy(cell_ids)
    idx_t_train = torch.from_numpy(train_idx.astype(np.int64))
    train_ds = TensorDataset(idx_t_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    emb_val = emb_t[val_idx].to(device)
    cell_val = cell_t[val_idx].to(device)
    lats_val = lats[val_idx]
    lons_val = lons[val_idx]
    centroid_lats = centroids[:, 0]
    centroid_lons = centroids[:, 1]

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_top1": [], "val_top5": [], "val_median_km": []}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_n = 0
        for (batch_idx,) in train_loader:
            idx_np = batch_idx.numpy()
            x = emb_t[batch_idx].to(device)
            logits = model(x)
            if smoothing is not None:
                soft = torch.from_numpy(smoothing.soft_targets(idx_np, args.tau_km)).to(device)
                loss = soft_cross_entropy(logits, soft)
            else:
                y = cell_t[batch_idx].to(device)
                loss = torch.nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            epoch_loss_sum += float(loss.item()) * bs
            epoch_n += bs
        scheduler.step()
        train_loss = epoch_loss_sum / max(1, epoch_n)

        model.eval()
        with torch.no_grad():
            val_logits = model(emb_val)
            if smoothing is not None:
                soft_val = torch.from_numpy(
                    smoothing.soft_targets(val_idx.astype(np.int64), args.tau_km)
                ).to(device)
                val_loss = soft_cross_entropy(val_logits, soft_val).item()
            else:
                val_loss = torch.nn.functional.cross_entropy(val_logits, cell_val).item()
            top1 = (val_logits.argmax(dim=-1) == cell_val).float().mean().item()
            top5 = (
                (val_logits.topk(5, dim=-1).indices == cell_val[:, None]).any(dim=-1).float().mean().item()
            )
            preds_np = val_logits.argmax(dim=-1).cpu().numpy()
            pred_lat = centroid_lats[preds_np]
            pred_lon = centroid_lons[preds_np]
            km_err = haversine_km(pred_lat, pred_lon, lats_val, lons_val)
            median_km = float(np.median(km_err))

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_top1"].append(top1)
        history["val_top5"].append(top5)
        history["val_median_km"].append(median_km)
        print(
            f"epoch {epoch+1:02d}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f} "
            f"| top1 {top1:.3f} | top5 {top5:.3f} | median_km {median_km:.1f}"
        )

    save_classifier(
        model.cpu(),
        args.out,
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        num_cells=num_cells,
        dropout=args.dropout,
        tau_km=args.tau_km,
        cell_centroids_latlon=centroids,
    )
    print(f"[train] saved to {args.out}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="val")
    axes[0].set_title("loss"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["epoch"], history["val_top1"], marker="o", label="top-1")
    axes[1].plot(history["epoch"], history["val_top5"], marker="o", label="top-5")
    axes[1].set_title("val accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(history["epoch"], history["val_median_km"], marker="o")
    axes[2].set_title("val median km error"); axes[2].set_xlabel("epoch"); axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    Path(args.curves).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.curves, dpi=120)
    plt.close(fig)
    print(f"[train] saved curves to {args.curves}")


if __name__ == "__main__":
    main()
