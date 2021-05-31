
"""
Driverless AIサーバ上にデプロイしたモデルを用いたスコアリング
spinboxから入力を受け取った1行（x1,x2,x3,x4）をPostする
"""

from h2o_wave import Q, main, app, ui, data
import requests
import json

api_point = "http://34.230.47.166:9090/model/score"    # 環境に合わせ、ここを変更

# 各Inputの入力範囲（spinboxで入力範囲の制約に用いる）
x1_min, x1_max = -10.0, 10.0
x2_min, x2_max = -10.0, 10.0
x3_min, x3_max = -10.0, 10.0
x4_min, x4_max = -10.0, 10.0


def scoring(x1: float, x2: float, x3: float, x4: float) -> str:
    """ Driverless AI scoring apiの呼び出し
    """
    headers = {'Content-Type': 'application/json'}
    data = {"fields":["x1", "x2", "x3", "x4"], "rows":[[str(x1), str(x2), str(x3), str(x4)]]}
    response = requests.post(api_point, headers=headers, data=json.dumps(data))
    #print(response.json())
    return response.json()['score'][0][0]


@app('/scoring')
async def serve(q: Q):
    print("q.args --> ", q.args)   # ブラウザからのアクション情報を確認したい
    if q.args.do_scoring:    # "Scoring"ボタンが押された場合
        res = scoring(q.args.x1, q.args.x2, q.args.x3, q.args.x4)
        q.page['card_scoring'] = ui.form_card(
            box='1 1 4 5',
            items=[
                ui.text_m('Input: X1={}, X2={}, X3={}, X4={}'.format(q.args.x1, q.args.x2, q.args.x3, q.args.x4)),
                ui.text_m('Scoring Result: {}'.format(res)),
                ui.button(name='back', label='Back', primary=True),
            ],
        )
    else:    # 初期画面
        q.page['card_scoring'] = ui.form_card(
            box='1 1 4 5',
            items=[
                ui.text_xl('スコアリングApp'),
                ui.separator(),
                ui.text_l('Driverless AIサーバ上にデプロイしたモデルをスコアリング'),
                ui.text_m('API Point: {}'.format(api_point)),
                ui.separator(),
                ui.spinbox(name='x1', label='X1 [{},{}]'.format(x1_min,x1_max), min=x1_min, max=x1_max, step=0.1),
                ui.spinbox(name='x2', label='X2 [{},{}]'.format(x2_min,x2_max), min=x2_min, max=x2_max, step=0.1),
                ui.spinbox(name='x3', label='X3 [{},{}]'.format(x3_min,x3_max), min=x3_min, max=x3_max, step=0.1),
                ui.spinbox(name='x4', label='X4 [{},{}]'.format(x4_min,x4_max), min=x4_min, max=x4_max, step=0.1),
                ui.button(name='do_scoring', label='Scoring', primary=True),
            ],
        )
    await q.page.save()
