"""
[2021/2/23]
PageにCardsを配置し、カード内に色んなアイテムを配置する
テクストボックスやボタンなどの動き（ブラウザからのアクション）を理解する
"""

from h2o_wave import Q, main, app, ui

@app('/layout')
async def serve(q: Q):
    print("q.args --> ", q.args)   # ブラウザからのアクション情報を確認したい。

    q.page['header'] = ui.header_card(    # これはCardです。header_cardを利用すると、Appの見栄えが良くなります。
        box='1 1 9 1',    # x座標 y座標 幅 高さ
        title='レイアウトとアクションの学習',
        subtitle='これはデモAppです。',
    )

    q.page['card_side'] = ui.form_card(    # これはCardです。form_cardはいろんな要素を縦に並べられます。
        box='1 2 3 8',
        items=[
            ui.text_xl('こんにちは。(text_xl)'),    # 表示するだけ
            ui.text_l('こんにちは。(text_l)'),      # 表示するだけ
            ui.text_m('こんにちは。(text_m)'),      # 表示するだけ
            ui.text_s('こんにちは。(text_s)'),      # 表示するだけ
            ui.text_xs('こんにちは。(text_xs)'),    # 表示するだけ
            ui.textbox(name='textbox1', label='Text Box（必須）です。', value='なんか入力', required=True),
            ui.textbox(name='textbox2', label='Text Boxです。', value='なんか入力'),
            ui.spinbox(name='spinbox1', label='Spin Boxです。', min=-5, max=5, step=0.1),
            ui.button(name='button1', label='Button1です。', primary=True),
            ui.button(name='button2', label='Button2です。', primary=True),
            ui.text_m('（ボタンを押すと入力情報などがWaveサーバに送られます。）'),      # 表示するだけ
        ]
    )

    q.page['card_main'] = ui.form_card(    # これはCardです。
        box='4 2 6 5',
        items=[
            ui.text_xl('ここに分析結果とかいろいろ表示すると良いと思います。'),      # 表示するだけ
            ui.separator('以下、アクション情報'),
            ui.text_m( 'q.args.textbox1には「{}」が入っています。'.format(str(q.args.textbox1))),
            ui.text_m( 'q.args.textbox2には「{}」が入っています。'.format(str(q.args.textbox2))),
            ui.text_m( 'q.args.spinbox1には「{}」が入っています。'.format(str(q.args.spinbox1))),
            ui.text_m( 'q.args.button1には「{}」が入っています。(Trueだとこのボタンが押されたと言うことです。)'.format(str(q.args.button1))),
            ui.text_m( 'q.args.button2には「{}」が入っています。(Trueだとこのボタンが押されたと言うことです。)'.format(str(q.args.button2))),
        ]
    )

    await q.page.save()
