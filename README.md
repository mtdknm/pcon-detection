# P-con Detection

コンクリート壁面画像からPコン跡の有無を判定するCNNモデルです。

## 実行環境

- Windows
- Python 3.12.0
- PyTorch 2.11.0
- Visual Studio Code
- CPU（最大4スレッド、GPU不使用）

その他のライブラリはrequirements.txtに記載しています。

## 使用データセット

SDNET2018

https://digitalcommons.usu.edu/all_datasets/48/

ハッカソンで指定された壁面画像と、運営から提供されたラベルを使用しています。外部データや外部の学習済みモデルは使用していません。

配布データはこのリポジトリには含めていません。

## ファイル構成

| ファイル名 | 内容 |
|---|---|
| train.py | CNNモデルの構築・学習・予測 |
| train_aug.py | データ拡張を強化したモデル（V13） |
| cross_validation.py | 3分割交差検証 |
| error_analysis.py | 誤判定画像の抽出 |
| requirements.txt | 使用ライブラリ |

## 実行方法

### 1. ライブラリのインストール

`python -m pip install -r requirements.txt`

### 2. データの準備

以下のデータを実行フォルダに配置してください。

- SDNET2018/
- train_label.csv
- test.txt

### 3. 学習・予測

`python train_aug.py --epochs 10 --size 96 --output submission_v13.csv`

実行後、以下のファイルが生成されます。

- submission_v13.csv：提出用CSV
- submission_v13.pt：学習済みモデル

## 学習設定

| 項目 | 設定 |
|---|---|
| 学習画像数 | 8,063枚 |
| 検証画像数 | 2,016枚 |
| 評価画像数 | 8,059枚 |
| 画像サイズ | 96×96 |
| エポック数 | 10 |
| バッチサイズ | 48 |
| 最適化手法 | AdamW |
| 乱数シード | 42 |

## 誤判定分析の実行方法

以下のコマンドを順番に実行してください。

`python cross_validation.py`

`python error_analysis.py`

error_analysisフォルダに分析結果が出力されます。

## 実験結果

| 項目 | 結果 |
|---|---:|
| Public F1スコア | 0.8045977011494253 |
| 検証F1スコア | 0.9189 |
| 判定閾値 | 0.6627 |
| Pコン跡ありの予測数 | 155枚 |


実行環境によって結果が変動する可能性があります。