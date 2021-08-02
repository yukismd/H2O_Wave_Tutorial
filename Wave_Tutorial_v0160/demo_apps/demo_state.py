"""
[2021/8/1]
Stateを理解する
"""

from h2o_wave import Q, main, app, ui

app_info = []
user_info = []
client_info = []

@app('/state')
async def serve(q: Q):

    if q.args.button:
        q.app.textbox.append(q.args.textbox)
        q.user.textbox.append(q.args.textbox)
        q.client.textbox.append(q.args.textbox)
    else:
        q.app.textbox = []
        q.user.textbox = []
        q.client.textbox = []  

    print("q.args --> ", q.args)   # ブラウザからのアクション情報を確認したい。
    print("q.app --> ", q.app)   # Check app status (app level)
    print("q.user --> ", q.user)   # Check user status (browser level)
    print("q.client --> ", q.client) # Check client status (browser tab level)

    q.page['header'] = ui.header_card(    # これはCardです。header_cardを利用すると、Appの見栄えが良くなります。
        box='1 1 9 1',    # x座標 y座標 幅 高さ
        title='Stateの学習',
        subtitle='これはデモAppです。',
    )

    q.page['card_side'] = ui.form_card(    # これはCardです。form_cardはいろんな要素を縦に並べられます。
        box='1 2 3 8',
        items=[
            ui.textbox(name='textbox', label='情報を入力', value='保持する情報'),
            ui.button(name='button', label='情報の送付', primary=True),
            ui.text_m('（ボタンを押すと入力情報などがWaveサーバに送られます。）'),      # 表示するだけ
        ]
    )

    q.page['card_main'] = ui.form_card(    # これはCardです。
        box='4 2 6 5',
        items=[
            ui.text_xl('アクション情報'),      # 表示するだけ
            ui.text_l( 'q.args.textboxには「{}」が入っています。'.format(str(q.args.textbox))),
            ui.text_l( 'q.app.textboxには「{}」が入っています。'.format(str(q.app.textbox))),
            ui.text_l( 'q.user.textboxには「{}」が入っています。'.format(str(q.user.textbox))),
            ui.text_l( 'q.client.textboxには「{}」が入っています。'.format(str(q.client.textbox))),
            ui.text_m( 'q.args.buttonには「{}」が入っています。(Trueだとボタンが押されたと言うことです。)'.format(str(q.args.button))),
        ]
    )

    await q.page.save()
