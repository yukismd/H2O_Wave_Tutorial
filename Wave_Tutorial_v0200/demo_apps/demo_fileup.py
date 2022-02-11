"""
[2021/2/22]
Waveサーバへのデータのアップロードとアップされたデータの表示、ダウンロード

参考：
https://h2oai.github.io/wave/docs/examples/file-upload
https://h2oai.github.io/wave/docs/examples/upload/
https://h2oai.github.io/wave/docs/examples/upload-download
"""

import os
from h2o_wave import main, app, Q, ui
import pandas as pd   # App上でデータハンドリングする場合は、pandasを利用

def display_dataframe(df: pd.DataFrame, n_rows=10) -> str:
    """ 
    pandas.DataFrameの表示
    ui.textへ渡すstrを生成
    """
    def make_markdown_row(values):
        return f"| {' | '.join([str(x) for x in values])} |"
    def make_markdown_table(fields, rows):
        head = "(先頭{}行の表示)\n".format(n_rows)
        return head + '\n'.join([
            make_markdown_row(fields),
            make_markdown_row('---' * len(fields)),
            '\n'.join([make_markdown_row(row) for row in rows]),
        ])
    return make_markdown_table(
        fields=df.head(n_rows).columns.tolist(),
        rows=df.head(n_rows).values.tolist()
        )


@app('/fileup')
async def serve(q: Q):

    if q.args.file_upload:    # データがアップロードされた場合の画面（”Upload!”ボタンが押された場合）
        server_path = q.args.file_upload[0]   # アップされたWaveサーバ上のファイル
        local_path = await q.site.download(server_path, path='.')  # Waveサーバ上のデータをApp実行パス上にロード、そのパスの取得(path: download path)
        df = pd.read_csv(local_path)   # pandas.DataFrameとして、App実行パス上のデータをAppへ読み込む
        
        df['Added_Column'] = 'ADDED!'   # App上でdfの加工を実施
        modified_data_name = 'modified_data.csv'   # 加工後のデータ名
        df.to_csv(modified_data_name, index=False)   # 加工済みデータをローカルに一旦保存
        server_dl_path = await q.site.upload([modified_data_name])   # ブラウザからダウンロードするため、サーバにアップ。アップロードしたサーバ上パスを取得

        q.page['example'] = ui.form_card(
            box='1 1 4 6', 
            items=[
                ui.text_xl("ファイルがサーバにアップされました。"),
                ui.text('アップされたWaveサーバ上のファイル：{}'.format(server_path)),  # ”/_f/43e93aaf-90d9-4cfb-ad6c-5ef6bca1e987/your_data.csv”などと表記
                ui.text('App実行ローカル上にDLされたファイル：{}'.format(local_path)),
                ui.text('サイズ(bytes)：{}'.format(os.path.getsize(local_path))),
                ui.text('データシェープ：{}行、{}列'.format(df.shape[0], df.shape[1])),
                ui.text(display_dataframe(df)),   # pandas.DataFrameをApp上で表示
                ui.link(label="元データのダウンロード", download=True, path=server_path, button=True),
                ui.link(label="加工データのダウンロード", download=True, path=server_dl_path[0], button=True),
                ui.button(name='show_upload', label='Back', primary=True),
                ]
            )
        
        os.remove(local_path)    # App実行パス上のデータの削除
        os.remove(modified_data_name)    # App実行パス上のデータの削除

    else:    # 初期画面
        q.page['example'] = ui.form_card(
            box='1 1 4 6',
            items=[
                ui.text_xl("ファイルをアップしましょう。"),
                ui.file_upload(name='file_upload', 
                            label='Upload!', 
                            multiple=True,
                            file_extensions=['csv'],   # 許可するファイル形式
                            max_file_size=10, 
                            max_size=15
                )    # データを指定し、ボタンを押すとWaveサーバにデータがアップされる（wavedのパス上のdataフォルダ）
            ]
        )
    
    print("q.args --> ", q.args)   # ブラウザからのアクション情報を確認したい。
    await q.page.save()

