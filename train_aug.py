
import random
import torch
import torch.nn.functional as F
import train


class StrongImages(train.Images):

    def __getitem__(self, i):

        # 従来のデータ拡張
        result = super().__getitem__(i)

        if not self.augment:
            return result

        x, label = result
        h, w = x.shape[-2:]

        # ① 縦横比・大きさの変更
        if random.random() < 0.5:

            scale_h = random.uniform(0.65, 1.0)
            scale_w = random.uniform(0.65, 1.0)

            nh = max(1, int(h * scale_h))
            nw = max(1, int(w * scale_w))

            x = F.interpolate(
                x.unsqueeze(0),
                size=(nh, nw),
                mode="bilinear",
                align_corners=False
            )

            # 画像を切り捨てずに余白を補う
            left = random.randint(0, w - nw)
            right = w - nw - left
            top = random.randint(0, h - nh)
            bottom = h - nh - top

            x = F.pad(
                x,
                (left, right, top, bottom),
                mode="replicate"
            ).squeeze(0)

        # ② コントラストの変更
        if random.random() < 0.5:

            mean = x.mean(
                dim=(1, 2),
                keepdim=True
            )

            contrast = random.uniform(0.5, 1.3)
            x = (x - mean) * contrast + mean

        # ③ 明るさの変更
        if random.random() < 0.3:
            x = x + random.uniform(-0.15, 0.15)

        # ④ 軽いぼかし
        if random.random() < 0.2:

            blurred = F.avg_pool2d(
                F.pad(
                    x.unsqueeze(0),
                    (1, 1, 1, 1),
                    mode="replicate"
                ),
                kernel_size=3,
                stride=1
            ).squeeze(0)

            strength = random.uniform(0.2, 0.7)
            x = x * (1 - strength) + blurred * strength

        # ⑤ 少量のノイズ
        if random.random() < 0.2:
            x = x + torch.randn_like(x) * 0.025

        x = x.clamp(-1, 1)

        return x, label


if __name__ == "__main__":
    train.Images = StrongImages
    train.main()
