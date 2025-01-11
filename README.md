# スケートボーダー重心位置予測チャレンジ 1位ソリューション

## 概要

このリポジトリは、NEDO Challenge, Motion Decoding Using Biosignalsで実施された[スケートボーダー重心位置予測チャレンジ](https://signate.jp/competitions/1430)で1位を獲得したソリューションを公開しています。  
スケートボーダーの筋電位(EMG)からスケートボーダーの重心速度を予測します。アプローチの詳細は[こちら](docs/report.pdf)を参照ください。

## データセット

コンペティションで使用したデータセットは、コンペティションの参加者のみが入手可能になっています。詳しくは[SIGNATE](https://signate.jp)にお問い合わせください。

## セットアップ

このプロジェクトをローカル環境で実行するための手順は以下の通りです。

1. リポジトリをクローンします。
    ```bash
    git clone https://github.com/kimpara3/skateboarder-velocity.git
    cd skateboarder-velocity
    ```
2. 必要なパッケージをインストールします。
    ```bash  
    pip install -r requirements.txt
    ```
3. データセットをダウンロードし、`train.mat`,`test.mat`及び`reference.mat`を同一ディレクトリに配置します。

## 実行方法

前処理、学習、推論を`main.ipynb`に記述してあります。


## ライセンス

このプロジェクトは、MIT Licenseの下でライセンスされています。
