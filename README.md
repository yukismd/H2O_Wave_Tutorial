# H2O_Wave_Tutorial

## H2O Waveチュートリアル
Wave 0.16.0 - コード:[Wave_Tutorial_v0160](Wave_Tutorial_v0160/), 資料：[Wave-GettingStarted_ver20210730.pdf](Wave-GettingStarted_ver20210730.pdf)  
Wave 0.17.0 - コード:[Wave_Tutorial_v0170](Wave_Tutorial_v0170/), 資料：[Wave-GettingStarted_ver20210908.pdf](Wave-GettingStarted_ver20210908.pdf)  
Wave 0.20.0 - コード:[Wave_Tutorial_v0200](Wave_Tutorial_v0200/), 資料：[Wave-GettingStarted_ver20210908.pdf](Wave-GettingStarted_ver220209.pdf)

***

### 実行方法
#### フォルダ構成
```
Wave_Tutorial_v<Waveバージョン>
 ├── requirements.txt  ... Python実行環境へインストールするライブラリ
 ├── demo_app/         ... 学習用デモアプリ
 ├── app.py            ... Batch Scoring App
 ├── sample_data/      ... サンプルデータ
 └── scoring_data/     ... Batch Scoring Appのスコアリングデータ保存先（Batch Scoring App実行の際scoring_dataとしてディレクトリを作成しておく）
```
#### アプリの実行（Mac OS環境）
[環境構築](https://wave.h2o.ai/docs/installation)
  
Python実行環境の準備と各アプリの実行（チュートリアルフォルダ内で実行）
```bash
Wave_Tutorial_v0160 % source v0160/bin/activate                                   (1)
(v0160) Wave_Tutorial_v0160 % python -V
Python 3.8.10
(v0160) Wave_Tutorial_v0160 % pip install -r requirements.txt                     (2)
(v0160) Wave_Tutorial_v0160 % wave run demo_apps/demo_hello_app.py                (3)
...
```
1. (1) Python仮想環境のアクティベート（'v0160'としてチュートリアルフォルダ内で環境作成済み）  
2. (2) 必要Pythonパッケージのインストール（初回のみ実施）  
3. (3) アプリの実行  
4. (3)の実行後、ブラウザからアクセス  
![hello app](./img/hello_app.png)

***
## Batch Scoring App
#### アプリの起動
```bash
(v0160) Wave_Tutorial_v0160 % wave run app.py
```
#### アプリの利用
![app1](./img/app_1.png)
![app2](./img/app_2.png)

***
### [H2O Wave 公式ドキュメント](https://wave.h2o.ai/)
