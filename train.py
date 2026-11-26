import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_curve, f1_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='SDNET2018')
    p.add_argument('--train', default='train_label.csv')
    p.add_argument('--test', default='test.txt')
    p.add_argument('--output', default='submission.csv')
    p.add_argument('--epochs', type=int, default=0, help='0: automatic')
    p.add_argument('--size', type=int, default=160)
    p.add_argument('--batch', type=int, default=48)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def clean_path(p):
    return str(p).strip().replace('\\', '/')


def source_group(p):
    # Example: W/CW/7069-3.jpg -> 7069. Ignore crack/non-crack directory.
    return Path(clean_path(p)).stem.rsplit('-', 1)[0]


def read_train(path):
    df = pd.read_csv(path, header=None, dtype=str, encoding='utf-8-sig')
    if df.shape[1] != 2:
        raise ValueError('Expected two columns in train_label.csv: image path, 0/1')
    y = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    if len(y) and pd.isna(y.iloc[0]):
        df = df.iloc[1:]
        y = y.iloc[1:]
    if not y.isin([0, 1]).all():
        raise ValueError('Unexpected training label; expected 0 or 1')
    return df.iloc[:, 0].map(clean_path).to_numpy(), y.to_numpy(dtype=np.int64)


def find_root(root, example):
    root = Path(root)
    if (root / example).is_file():
        return root
    if root.is_dir():
        for w in root.rglob('W'):
            if (w.parent / example).is_file():
                return w.parent
    raise FileNotFoundError(f'Cannot find {example} in {root}. Set --root to the folder containing W.')


class Images(Dataset):
    def __init__(self, root, paths, labels=None, size=160, augment=False):
        self.root, self.paths, self.labels = root, paths, labels
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / clean_path(self.paths[i])) as f:
            im = f.convert('RGB').resize((self.size, self.size), Image.Resampling.BILINEAR)
        if self.augment:
            if random.random() < 0.5:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            turns = random.randrange(4)
            if turns:
                im = im.transpose([None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270][turns])
            if random.random() < 0.5:
                im = ImageEnhance.Brightness(im).enhance(random.uniform(0.85, 1.15))
                im = ImageEnhance.Contrast(im).enhance(random.uniform(0.85, 1.15))
        x = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1).copy()) / 127.5 - 1.0
        if self.labels is None:
            return x
        return x, torch.tensor(self.labels[i], dtype=torch.float32)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(),
                                 nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.net(x) + self.skip(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 24, 5, 2, 2, bias=False), nn.BatchNorm2d(24), nn.SiLU(),
                                      ConvBlock(24, 32, 2), ConvBlock(32, 48, 2), ConvBlock(48, 80, 2), ConvBlock(80, 112))
        self.head = nn.Sequential(nn.Linear(224, 96), nn.SiLU(), nn.Dropout(0.2), nn.Linear(96, 1))

    def forward(self, x):
        x = self.features(x)
        x = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return self.head(x).squeeze(1)


def group_split(paths, y, seed):
    groups = np.array([source_group(p) for p in paths])
    if len(set(groups)) < 2:
        raise ValueError('Cannot split by source photo: check file naming in source_group()')
    splitter = GroupShuffleSplit(n_splits=64, test_size=0.2, random_state=seed)
    best = None
    for tr, va in splitter.split(paths, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        score = abs(y[tr].mean() - y.mean()) + abs(y[va].mean() - y.mean()) + abs(len(va)/len(y) - 0.2)
        if best is None or score < best[0]:
            best = (score, tr, va)
    if best is None:
        raise ValueError('No group split containing both classes in both sets. Check positive labels and source groups.')
    return best[1], best[2]


@torch.inference_mode()
def predict(model, loader, device, labeled):
    model.eval()
    probs = []
    for batch in loader:
        x = (batch[0] if labeled else batch).to(device, non_blocking=True)
        # Same test-time augmentation for validation and test.
        p = (model(x).sigmoid() + model(x.flip(-1)).sigmoid()) * 0.5
        probs.append(p.cpu().numpy())
    return np.concatenate(probs)


def choose_threshold(y, probs):
    precision, recall, thresholds = precision_recall_curve(y, probs)
    score = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.argmax(score))
    return float(thresholds[idx]), float(score[idx])


def main():
    a = args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    epochs = a.epochs or (12 if device.type == 'cuda' else 4)
    print(f'Device: {device}; epochs: {epochs}', flush=True)
    paths, y = read_train(a.train)
    test_original = [line.strip() for line in Path(a.test).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not len(paths) or not len(test_original):
        raise ValueError('Empty training or test list')
    root = find_root(a.root, paths[0])
    tr, va = group_split(paths, y, a.seed)
    print(f'Train: {len(tr)} (positive {y[tr].sum()}); validation: {len(va)} (positive {y[va].sum()}); test: {len(test_original)}', flush=True)
    print(f'Image root: {root}', flush=True)
    def loader(dataset, shuffle=False):
        return DataLoader(dataset, batch_size=a.batch, shuffle=shuffle, num_workers=a.workers, pin_memory=device.type == 'cuda')
    train_loader = loader(Images(root, paths[tr], y[tr], a.size, True), True)
    val_loader = loader(Images(root, paths[va], y[va], a.size))
    test_loader = loader(Images(root, test_original, size=a.size))
    model = Model().to(device)
    weight = np.clip(np.sqrt((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())), 1., 25.)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(weight), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    best_f1 = -1.
    best = None
    checkpoint = Path(a.output).with_suffix('.pt')
    for ep in range(epochs):
        model.train()
        total_loss = 0.
        for step, (x, labels) in enumerate(train_loader, 1):
            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
            if step % 50 == 0:
                print(f'Epoch {ep+1}/{epochs}: batch {step}/{len(train_loader)}', flush=True)
        scheduler.step()
        probs = predict(model, val_loader, device, True)
        threshold, f1 = choose_threshold(y[va], probs)
        print(f'Epoch {ep+1}/{epochs} loss={total_loss/len(tr):.4f} val_F1={f1:.4f} threshold={threshold:.4f}', flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_threshold = threshold
            torch.save({'model': best, 'threshold': best_threshold, 'size': a.size, 'val_f1': best_f1}, checkpoint)
    model.load_state_dict(best)
    probs = predict(model, test_loader, device, False)
    pred = (probs >= best_threshold).astype(int)
    pd.DataFrame({0: test_original, 1: pred}).to_csv(a.output, index=False, header=False)
    print(f'Best validation F1: {best_f1:.4f}; threshold: {best_threshold:.4f}', flush=True)
    print(f'Saved {a.output}: {len(pred)} rows, {pred.sum()} positives', flush=True)


if __name__ == '__main__':
    main()
