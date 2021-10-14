"""
Driverless AIへの接続テンプレート
1. DAI接続情報（URL, User ID, Password）の入力
2. モデル（Experiment）の選択
3. 選択されたモデル情報の表示
"""

from h2o_wave import main, app, Q, ui, on, handle_on, data
from driverlessai import Client


@app('/dai')
async def serve(q: Q):

    # Set up the application
    if not q.client.initialized:
        await initialize_app_for_new_client(q)

    # Handle all button clicks and file upload
    await handle_on(q)

    print("q.args --> ", q.args)   # ブラウザからのアクション情報を確認したい
    print("q.client --> ", q.args)
    # Save content to the browser
    await q.page.save()


async def initialize_app_for_new_client(q: Q):
    """ 初期設定（metaカード、ヘッダー）
    """

    if not q.user.initialized:
        await initialize_app_for_new_user(q)

    q.page['meta'] = ui.meta_card(
        box='',
        title='DAI Template',
        theme='light',
    )

    # Adding ui elements
    q.page['header'] = ui.header_card(
        box='1 1 11 1',
        title='Driverless AI Connection Sample',
        subtitle='Driverless AIへの接続テンプレート',
    )

    q.client.dai_connection = False
    q.client.model_selected = False
    q.client.data_uploaded = False
    render_sidebar_content(q)

    q.client.initialized = True


async def initialize_app_for_new_user(q: Q):
    """初期化フラグ（Userレベル）"""

    if not q.app.initialized:
        await initialize_app(q)

    q.user.initialized = True


async def initialize_app(q: Q):
    """初期化フラグ（Appレベル）"""
    q.app.initialized = True


def render_sidebar_content(q: Q):
    """ 
    sidebarコンテンツの表示振り分け
    """

    if not q.client.dai_connection:    # DAIに接続されてない場合（DAI接続設定）
        sidebar_items = get_dai_configure_items(q)
    elif not q.client.model_selected:  # モデルが選択されていない場合（モデル選択）
        sidebar_items = get_model_selection_items(q)

    q.page['sidebar'] = ui.form_card(
        box='1 2 3 8',
        items=[
                  ui.stepper(name='almost-done-stepper', items=[
                      ui.step(label='Connect', done=q.client.dai_connection),
                      ui.step(label='Select', done=q.client.model_selected),
                  ])
              ] + sidebar_items
    )


@on('dai_connect_button')    # DAI接続ボタン(Connect)のクリック後すぐに実施される
async def handle_dai_connection(q: Q):

    q.user.dai_url = q.args.dai_url
    q.user.dai_username = q.args.dai_username
    q.user.dai_password = q.args.dai_password

    _, q.client.error = create_dai_connection(q)
    if q.client.error is None:
        q.client.dai_connection = True

    render_sidebar_content(q)


@on('select_model_button')    # モデル選択ボタン（Select）のクリック後すぐに実施される
async def handle_model_selection(q: Q):

    q.client.experiment_key = q.args.experiment_dropdown    # DAIモデルID
    q.client.model_selected = True
    render_center_content(q)


def get_dai_configure_items(q: Q):
    """ 
    DAIに接続する初期画面（sidebarの要素）
    dai_connect_button: [Connect]ボタンが押されるとTrue
    """

    if q.client.error is not None:
        dai_message = ui.message_bar(type='error', text=f'Connection Failed: {q.client.error}')
    elif q.client.dai_conn is None:
        dai_message = ui.message_bar(type='warning', text='This app is not connected to DAI!')
    else:
        dai_message = ui.message_bar(type='success', text='This app is connected to DAI!')

    dai_connection_items = [
        ui.separator('Connect to DAI'),
        dai_message,
        ui.textbox(name='dai_url', label='Driverlss AI URL', value=q.user.dai_url, required=True),
        ui.textbox(name='dai_username', label='Driverless AI Username', value=q.user.dai_username, required=True),
        ui.textbox(name='dai_password', label='Driverless AI Password', value=q.user.dai_password, required=True, password=True),
        ui.buttons([ui.button(name='dai_connect_button', label='Connect', primary=True)], justify='center')
    ]
    return dai_connection_items


def get_model_selection_items(q: Q):
    """
    DAI接続後のモデル選択画面（sidebarの要素）
    experiment_dropdown（DAIモデルID）のボタン情報
    select_model_button: [Select]ボタンが押されるとTrue
    """

    dai, error = create_dai_connection(q)

    ui_choice_experiments = [ui.choice(d.key, d.name) for d in dai.experiments.list()]
    model_selection_items = [
        ui.separator('Select a Model'),
        ui.dropdown(name='experiment_dropdown', label='Driverless AI Models', required=True,
                    choices=ui_choice_experiments, value=q.client.experiment_key),
        ui.buttons([ui.button(name='select_model_button', label='Select', primary=True)], justify='center')
    ]
    return model_selection_items


def create_dai_connection(q):
    """ 
    DAIに接続しClientオブジェクトを返す
    http://docs.h2o.ai/driverless-ai/pyclient/docs/html/client.html
    """

    try:
        conn = Client(q.user.dai_url, q.user.dai_username, q.user.dai_password, verify=False)
        return conn, None
    except Exception as ex:
        return None, str(ex)


def render_center_content(q: Q):
    """
    モデル選択後のモデル情報の表示（sidebarの要素）
    """
    dai, error = create_dai_connection(q)
    experiment = dai.experiments.get(q.client.experiment_key)   # 選択されたExperimentオブジェクトの取得

    q.page['model_info'] = ui.form_card(
        box='4 2 8 5',
        items=[
            ui.text_l('選択されたモデルに関して'),
            ui.separator(),
            ui.text_m('--- 学習データのカラム名 ---'),
            ui.text_s(str(experiment.datasets['train_dataset'].columns)),
            ui.text_m('--- ターゲット変数名 ---'),
            ui.text_s(experiment.settings['target_column']),
            ui.text_m('--- 回帰（regression）or 分類（classification）---'),
            ui.text_s(experiment.settings['task']),
        ]
    )

