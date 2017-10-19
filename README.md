# Japanese Talk API

## Setup

1. python 2.7.9にする

```
pyenv local 2.7.9
```

1. 依存ライブラリのインストール

```
pip install -r requirements.txt
```

1. tornado/models/の下にvocab.binとjapanese_talk_api.chainermodelを置く（https://github.com/m-doi/chainer-char-rnn で作ったものをrenameして置く）

## Run

1. 起動する
```
./run_app_local.sh
```

1. 試してみる

```
http://localhost:8787/?q=宮島さんの
```
