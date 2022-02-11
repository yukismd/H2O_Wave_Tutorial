"""
[2021/2/26]
ルーティング（ブラウザからのアクションにしたがって表示を切り替える）の理解
"""

from h2o_wave import Q, main, app, ui

@app('/routing1')
async def serve(q: Q):
    print("q.args --> ", q.args)   # ブラウザからのアクション情報を確認したい。

    q.page['header'] = ui.header_card(    # このヘッダーは常に表示します。
        box='1 1 9 1',
        title='ルーティングの学習(1)',
        subtitle='これはデモAppです。',
    )

    if q.args.button_A:      # button_Aが押された場合実行。（q.args.button_A==True）
        print('-- button_Aが押された --')
        await do_button_A(q)

    elif q.args.button_B:    # button_Bが押された場合実行。（q.args.button_B==True）
        print('-- button_Bが押された --')
        await do_button_B(q)

    else:                    # 初期画面。button_Aもbutton_Bも押されていない。
        print('-- 初期画面です --')
        await initial_display(q)

    await q.page.save()

async def initial_display(q: Q):
    q.page['card_side'] = ui.form_card(
        box='1 2 3 8',
        items=card_side_items()
    )

    del q.page['card_main']    # Button Aの表示を削除（あってもなくても）

async def do_button_A(q: Q):
    """ button_Aが押された場合の表示
    """
    q.page['card_side'] = ui.form_card(
        box='1 2 3 8',
        items=card_side_items()
    )
    q.page['card_main'] = ui.form_card(
        box='4 2 6 5',
        items=[
            ui.text_xl('Button Aが押されました。'),
        ]
    )

async def do_button_B(q: Q):
    """ button_Bが押された場合の表示
    """
    q.page['card_side'] = ui.form_card(
        box='1 2 3 8',
        items=card_side_items() + 
            [ui.separator(), ui.text_xl('Button Bが押されました。'),]
    )
    del q.page['card_main']    # Button Aの表示を削除。（あってもなくても）

def card_side_items():
    return [ui.text_xl('押すボタンにより、表示を切り替えます。'),
            ui.button(name='button_A', label='Button A', primary=True),
            ui.button(name='button_B', label='Button B', primary=True),
            ui.button(name='button_back', label='戻る', primary=False),    # これが押されるとbutton_Aもbutton_BもFalseのアクションとなる
            ]
