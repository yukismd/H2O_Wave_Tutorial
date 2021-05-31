"""
Driverless AIサーバ上にデプロイしたモデルを用いたスコアリング
spinboxから入力を受け取った1行（x1,x2,x3,x4）をPostする
"""

from h2o_wave import Q, main, app, ui, data
import requests
import json
import os
import pandas as pd

api_point = "http://3.88.112.113:9090/model/score"    # 環境に合わせ、ここを変更


def display_dataframe(df: pd.DataFrame, n_rows=5) -> str:
    """ 
    pandas.DataFrameの表示
    ui.textへ渡すstrを生成
    """
    def make_markdown_row(values):
        return f"| {' | '.join([str(x) for x in values])} |"
    def make_markdown_table(fields, rows):
        return '\n'.join([
            make_markdown_row(fields),
            make_markdown_row('---' * len(fields)),
            '\n'.join([make_markdown_row(row) for row in rows]),
        ])
    return make_markdown_table(
        fields=df.head(n_rows).columns.tolist(),
        rows=df.head(n_rows).values.tolist()
        )

def batch_scoring(df: pd.DataFrame) -> list:
    """
    df.columns = ["x1","x2","x3","x4"]
    """
    df_str = df.astype(str)
    headers = {'Content-Type': 'application/json'}
    data = {"fields":["x1", "x2", "x3", "x4"], "rows":[[row[0],row[1],row[2],row[3]] for index,row in df_str.iterrows()]}
    response = requests.post(api_point, headers=headers, data=json.dumps(data))
    return response.json()['score']

@app('/scoring_batch')
async def serve(q: Q):
    print("q.args --> ", q.args)   # ブラウザからのアクション情報を確認したい
    if q.args.do_scoring:    # スコアリングボタン（do_scoring）が押された場合
        df = pd.read_csv(q.client.local_path)    # App実行パス上にスコアリング用データが存在することを想定
        res = batch_scoring(df)
        df_score = pd.DataFrame({"index":[i for i in range(len(res))], "score":[float(i[0]) for i in res]})
        df_score.to_csv("./result.csv", index=False)    # ローカルに仮保存
        download_path = await q.site.upload(["./result.csv"])   # Waveサーバにアップロードし、パスを取得
        os.remove("./result.csv")  # ローカルに仮保存したファイルを削除

        ## スコアリング結果、ダウンロード
        q.page['card_scoring'] = ui.form_card(
            box='1 1 4 3',
            items=[
                ui.text_l("スコアリング結果"),
                ui.separator(),
                ui.text('Scoring Result: {}'.format(res)),
                ui.link(label="結果のダウンロード", download=True, path=download_path[0]),#, button=True),
                ui.separator(),
                ui.button(name='back', label='Back to file upload', primary=False),
            ],
        )
        ## スコアリング結果のプロット
        q.page['card_scatter'] = ui.plot_card(
            box='1 4 4 5',
            title='スコアリング結果',
            data=data(
                fields=df_score.columns.tolist(),
                rows=df_score.values.tolist(),
                pack=True,    # if True, データを変更しない場合、圧縮しメモリ節約
            ),
            plot=ui.plot(
                marks=[ui.mark(
                    type='point',
                    x='=index', x_title='row index',
                    y='=score', y_title='score',
                    shape='circle',
                )]
            )
        )
        os.remove(q.client.local_path)    # App実行パス上のデータの削除sw
        q.client.remove_scatter = True    # ページ遷移した場合のq.page['card_scatter']の削除指示
    elif q.args.fileup:    # ファイルのアップロードが実施された場合のデータ確認画面
        server_path = q.args.fileup[0]  # アップされたファイルのサーバパス
        q.client.local_path = await q.site.download(server_path, '.')  # App実行パス上へのデータのダウンロードとそのパスの取得
        df = pd.read_csv(q.client.local_path)
        q.page['card_scoring'] = ui.form_card(
            box='1 1 4 6', 
            items=[
                ui.text_l("スコアリング用データ"),
                ui.separator(),
                ui.text_m('サイズ(bytes)：{}'.format(os.path.getsize(q.client.local_path))),
                ui.text_m('データシェープ：{}行、{}列'.format(df.shape[0], df.shape[1])),
                ui.text_m(display_dataframe(df)),
                ui.button(name='do_scoring', label='Scoring this data', primary=True),
                ui.button(name='show_upload', label='Back to file upload', primary=False),
            ]
        )
    else:    # 初期画面(ファイルのアップロード)
        if q.client.remove_scatter:    # スコアリング結果画面から遷移してきた場合、散布図のカードを削除
            del q.page['card_scatter']
        q.page['card_scoring'] = ui.form_card(
            box='1 1 4 6',
            items=[
                ui.text_xl('バッチスコアリングApp'),
                ui.separator(),
                ui.text_l('Driverless AIサーバ上にデプロイしたモデルをスコアリング'),
                ui.text_m('API Point: {}'.format(api_point)),
                ui.file_upload(name='fileup', 
                            label='Upload file', 
                            multiple=False,
                            file_extensions=['csv'],
                            max_file_size=10,
                            max_size=15
                    )
                ]
        )
    await q.page.save()