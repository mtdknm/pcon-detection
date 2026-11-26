
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix

from train import (
    Model, Images, read_train, find_root,
    source_group, predict, choose_threshold
)

torch.set_num_threads(4)
device = torch.device("cpu")

output = Path("error_analysis")
output.mkdir(exist_ok=True)

paths, y = read_train("train_label.csv")
root = find_root("SDNET2018", paths[0])

groups = np.array([
    source_group(p) for p in paths
])

splitter = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

probs = np.zeros(len(y))
fold_ids = np.zeros(len(y), dtype=int)

# 交差検証のモデルで予測
for fold, (tr, va) in enumerate(
    splitter.split(paths, y, groups), 1
):
    print(f"Analyzing Fold {fold}...", flush=True)

    ckpt = torch.load(
        f"cv_fold{fold}.pt",
        map_location=device
    )

    model = Model().to(device)
    model.load_state_dict(ckpt["model"])

    loader = DataLoader(
        Images(
            root, paths[va], y[va],
            ckpt["size"]
        ),
        batch_size=48
    )

    probs[va] = predict(
        model, loader, device, True
    )

    fold_ids[va] = fold

threshold, f1 = choose_threshold(y, probs)
pred = (probs >= threshold).astype(int)

# 判定結果を4種類に分類
category = np.select(
    [
        (y == 1) & (pred == 1),
        (y == 1) & (pred == 0),
        (y == 0) & (pred == 1)
    ],
    ["TP", "FN", "FP"],
    default="TN"
)

df = pd.DataFrame({
    "path": paths,
    "actual": y,
    "prediction": pred,
    "score": probs,
    "category": category,
    "fold": fold_ids,
    "source": groups
})

df.to_csv(
    output / "analysis.csv",
    index=False
)

print("OOF F1:", f1)
print("Threshold:", threshold)
print("Confusion matrix:")
print(confusion_matrix(y, pred))
print(df["category"].value_counts())

# 誤判定した画像を一覧画像にする
def make_gallery(rows, name):

    rows = rows.reset_index(drop=True)

    cols = 4
    per_page = 16
    cell_w = 260
    cell_h = 300

    for start in range(0, len(rows), per_page):

        page = rows.iloc[start:start+per_page]

        canvas = Image.new(
            "RGB",
            (cols * cell_w, 4 * cell_h),
            "white"
        )

        draw = ImageDraw.Draw(canvas)

        for j, (_, row) in enumerate(page.iterrows()):

            x = (j % cols) * cell_w
            y0 = (j // cols) * cell_h

            with Image.open(root / row["path"]) as im:
                im = im.convert("RGB")
                im = im.resize((256, 256))

            canvas.paste(im, (x, y0))

            draw.text(
                (x+3, y0+258),
                row["path"],
                fill="black"
            )

            draw.text(
                (x+3, y0+272),
                f"Score: {row['score']:.4f}",
                fill="black"
            )

            draw.text(
                (x+3, y0+285),
                f"Fold: {row['fold']}",
                fill="black"
            )

        canvas.save(
            output / f"{name}_{start//per_page+1:02}.jpg"
        )


# 見逃した画像
fn = df[df["category"] == "FN"]
fn = fn.sort_values("score")
make_gallery(fn, "FN")

# 誤検出した画像
fp = df[df["category"] == "FP"]
fp = fp.sort_values("score", ascending=False)
make_gallery(fp, "FP")

# 正しく検出した画像（比較用）
tp = df[df["category"] == "TP"]
tp = tp.sort_values("score", ascending=False)
make_gallery(tp.head(32), "TP")

print("Saved error analysis!")
