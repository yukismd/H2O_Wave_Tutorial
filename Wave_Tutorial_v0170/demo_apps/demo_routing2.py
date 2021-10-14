"""
[2021/2/26]
ルーティング（ブラウザからのアクションにしたがって表示を切り替える）の理解
on, handle_onを利用したルーティング（https://h2oai.github.io/wave/docs/routing/）
"""

from h2o_wave import Q, main, app, ui, on, handle_on

@app('/routing2')
async def serve(q: Q):
    print("q.args --> ", q.args)   # ブラウザからのアクション情報を確認したい。

    q.page['header'] = ui.header_card(    # このヘッダーは常に表示します。
        box='1 1 9 1',
        title='ルーティングの学習(2)',
        subtitle='これはデモAppです。',
    )

    if not q.client.initialized:    # q.clientはブラウザ単位で持つ情報です。
        print('-- 初期画面です（ここには最初の１回目しかきません） --')
        await initial_display(q)
    
    if q.args.button_back:
        print('-- 戻るボタンが押されても、初期画面に行きます --')
        await initial_display(q)

    await handle_on(q)

    await q.page.save()


async def initial_display(q: Q):
    q.page['card_side'] = ui.form_card(
        box='1 2 3 8',
        items=[ui.text_l('最初の画面です。')] + card_side_items()
    )
    
    del q.page['card_main']    # Button Aの表示を削除。（あってもなくても）
    
    q.client.initialized = True   # 初期化されたことを示すフラグです。（initializedとしていますが、何でも良いです。）


@on('button_A')    # Button Aが押されると直ぐに実行される。
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


@on('button_B')    # Button Bが押されると直ぐに実行される。
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
            ui.button(name='button_back', label='戻る', primary=False),
            ]
