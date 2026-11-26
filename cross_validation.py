
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from train import (
    Model, Images, read_train, find_root,
    source_group, predict, choose_threshold
)

# 学習設定
FOLDS = 3
EPOCHS = 10
SIZE = 96
BATCH = 48

device = torch.device("cpu")
torch.set_num_threads(4)

# データ読み込み
paths, y = read_train("train_label.csv")
root = find_root("SDNET2018", paths[0])

test_paths = [
    s.strip()
    for s in Path("test.txt").read_text(
        encoding="utf-8-sig"
    ).splitlines()
    if s.strip()
]

groups = np.array([source_group(p) for p in paths])

splitter = StratifiedGroupKFold(
    n_splits=FOLDS,
    shuffle=True,
    random_state=42
)

splits = list(splitter.split(paths, y, groups))

for tr, va in splits:
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
        raise ValueError("学習または検証データに正例・負例が不足しています")

oof_probs = np.zeros(len(y))
test_probs = []

# 3つのAIを学習
for fold, (tr, va) in enumerate(splits, 1):

    print(f"\n===== Fold {fold}/{FOLDS} =====", flush=True)

    random.seed(42 + fold)
    np.random.seed(42 + fold)
    torch.manual_seed(42 + fold)

    train_loader = DataLoader(
        Images(root, paths[tr], y[tr], SIZE, True),
        batch_size=BATCH,
        shuffle=True
    )

    val_loader = DataLoader(
        Images(root, paths[va], y[va], SIZE),
        batch_size=BATCH
    )

    test_loader = DataLoader(
        Images(root, test_paths, size=SIZE),
        batch_size=BATCH
    )

    model = Model().to(device)

    weight = np.clip(
        np.sqrt(
            (y[tr] == 0).sum()
            / max(1, (y[tr] == 1).sum())
        ),
        1, 25
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            float(weight), device=device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=0.0001
    )

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for x, labels in train_loader:

            x = x.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x)

        scheduler.step()

        print(
            f"Fold {fold} "
            f"Epoch {epoch+1}/{EPOCHS} "
            f"Loss={total_loss/len(tr):.4f}",
            flush=True
        )

    # 学習に使っていない画像で検証
    probs = predict(model, val_loader, device, True)
    oof_probs[va] = probs

    threshold, f1 = choose_threshold(y[va], probs)

    print(f"Fold {fold} F1: {f1:.4f}", flush=True)

    # 本番画像の予測
    test_probs.append(
        predict(model, test_loader, device, False)
    )

    torch.save({
        "model": model.state_dict(),
        "size": SIZE,
        "threshold": threshold
    }, f"cv_fold{fold}.pt")

# 全検証画像を使って判定閾値を決定
threshold, f1 = choose_threshold(y, oof_probs)

# 3モデルの予測を平均
avg_probs = np.mean(test_probs, axis=0)
pred = (avg_probs >= threshold).astype(int)

# 提出ファイル作成
pd.DataFrame({
    0: test_paths,
    1: pred
}).to_csv(
    "submission_cv.csv",
    index=False,
    header=False
)

print(f"\nOOF F1: {f1:.4f}")
print(f"Threshold: {threshold:.4f}")
print(f"Positives: {pred.sum()}")
print("Saved submission_cv.csv")
