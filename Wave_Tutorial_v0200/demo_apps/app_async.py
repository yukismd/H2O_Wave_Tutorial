from h2o_wave import main, app, Q, ui, on, handle_on, data
from driverlessai import Client

import pandas as pd
import numpy as np


@app('/app_async')
async def serve(q: Q):
    """This function will route the user based on how they have interacted with the application."""

    # Set up the application
    if not q.client.initialized:
        await initialize_app_for_new_client(q)

    # Handle all button clicks and file upload
    await handle_on(q)

    print("q.args --> ", q.args)   # ブラウザからのアクション情報の確認
    print("q.client --> ", q.client)
    print('#################### App Saved ####################')
    # Save content to the browser
    await q.page.save()


async def initialize_app_for_new_client(q: Q):
    """Setup this Wave application for each browser tab by creating a page layout and setting any needed variables"""
    print('>>>>> Initialization')

    if not q.user.initialized:
        await initialize_app_for_new_user(q)

    q.page['meta'] = ui.meta_card(
        box='',
        title='DAI Scoring App',
        theme='light',
    )

    # Adding ui elements
    q.page['header'] = ui.header_card(
        box='1 1 11 1',
        title='Driverless AI Scoring App',
        subtitle='Get predictions on new data from Driverless AI models.',
    )

    q.client.dai_connection = False
    q.client.model_selected = False
    q.client.data_uploaded = False
    await render_sidebar_content(q)

    q.client.initialized = True


async def initialize_app_for_new_user(q: Q):
    """初期化フラグ（Userレベル）"""
    if not q.app.initialized:
        await initialize_app(q)
    q.user.initialized = True


async def initialize_app(q: Q):
    """初期化フラグ（Appレベル）"""
    q.app.initialized = True


async def render_sidebar_content(q: Q):
    """ 
    サイドバーコンテンツの表示振り分け
    トップのConnect,Select,ScoreのStepperは常に表示
    """

    if not q.client.dai_connection:    # DAIに接続されてない場合（DAI接続設定）
        sidebar_items = get_dai_configure_items(q)
    elif not q.client.model_selected:  # モデルが選択されていない場合（モデル選択）
        sidebar_items = get_model_selection_items(q)
    elif not q.client.data_uploaded:   # データがアップロードされていない場合（データのアップロード）
        sidebar_items = get_batch_score_items(q)

    q.page['sidebar'] = ui.form_card(
        box='1 2 3 8',
        items=[
                  ui.stepper(name='almost-done-stepper', items=[
                      ui.step(label='Connect', done=q.client.dai_connection),
                      ui.step(label='Select', done=q.client.model_selected),
                      ui.step(label='Score', done=q.client.data_uploaded),
                  ])
              ] + sidebar_items
    )


@on('dai_connect_button')    # DAI接続ボタン（dai_connect_button）のクリック後すぐに実施される
async def handle_dai_connection(q: Q):
    print('>>>>> DAI Connection Done')

    q.user.dai_url = q.args.dai_url
    q.user.dai_username = q.args.dai_username
    q.user.dai_password = q.args.dai_password

    q.client.dai, q.client.error = create_dai_connection(q)   # q.client.dai -> driverlessai.Client object
    if q.client.error is None:
        q.client.dai_connection = True

    await render_sidebar_content(q)


@on('select_model_button')    # モデル選択ボタン（select_model_button）のクリック後すぐに実施される
async def handle_model_selection(q: Q):
    print('>>>>> Model Chosen')

    q.client.experiment_key = q.args.experiment_dropdown    # DAIモデルID
    q.client.model_selected = True
    await render_sidebar_content(q)


@on('file_upload')    # スコアリングデータアップロードボタン（file_upload）のクリック後すぐに実施される
async def handle_batch_scoring(q: Q):
    print('>>>>> Data Uploaded')

    q.client.batch_data_path = await q.site.download(url=q.args.file_upload[0], path='./scoring_data')  # App実行パス上へのデータのDLとDLパスの取得
    q.client.data_uploaded = True
    await render_center_content(q)


def get_dai_configure_items(q: Q):
    """ 
    DAIに接続する初期画面
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
    DAI接続後のモデル選択画面
    select_model_button: [Select]ボタンが押されるとTrue
    """

    #dai, error = create_dai_connection(q)

    ui_choice_experiments = [ui.choice(d.key, d.name) for d in q.client.dai.experiments.list()]
    model_selection_items = [
        ui.separator('Select a Model'),
        ui.dropdown(name='experiment_dropdown', label='Driverless AI Models', required=True,
                    choices=ui_choice_experiments, value=q.client.experiment_key),
        ui.buttons([ui.button(name='select_model_button', label='Select', primary=True)], justify='center')
    ]
    return model_selection_items


def get_batch_score_items(q: Q):
    """
    データのアップロード画面
    file_upload: [Score Data]ボタンが押されるとデータがWaveサーバにアップされ、（サーバ上仮想）パスが返る
    """

    #dai, error = create_dai_connection(q)
    experiment = q.client.dai.experiments.get(q.client.experiment_key)
    q.client.expected_columns = experiment.datasets['train_dataset'].columns   # 選択されたモデルの学習データのカラム名

    get_predictions_items = [
        ui.message_bar(type='info',
                       text=f'Expecting a csv with the following columns: {q.client.expected_columns}'),
        ui.file_upload(name='file_upload', label='Score Data', file_extensions=['csv'])
    ]
    return get_predictions_items


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


def get_dai_new_predictions(q):
    """ スコアリングの実施
    """

    # connect to DAI and get reference to the experiment
    #dai, error = create_dai_connection(q)
    experiment = q.client.dai.experiments.get(q.client.experiment_key)

    q.client.target_column = experiment.settings['target_column']   # Experiment情報：ターゲット変数名
    q.client.modeling_task = experiment.settings['task']    # Experiment情報：classification/regression
    print('-------------------------- Start Scoring --------------------------')
    data_to_predict = q.client.dai.datasets.create(q.client.batch_data_path, name='test_data_set', force=True)  # データをDAIにアップロード（Datasetクラス）
    dai_predictions = experiment.predict(data_to_predict, include_columns=data_to_predict.columns)   # スコアリングの実施
    data_to_predict.delete()
    print('-------------------------- End Scoring --------------------------')

    local_file_path = dai_predictions.download('.')    # スコアリング結果のダウンロード先（App実行上のパス）
    local_data_predictions = pd.read_csv(local_file_path)
    #print(local_data_predictions.head())

    return local_data_predictions


async def render_center_content(q: Q):
    """ データアップロード後の結果の表示
    """

    print('-------------------------- Start Scoring fn (get_dai_new_predictions) --------------------------')
    df = await q.run(get_dai_new_predictions, q)    # スコアリング結果をpandas.DataFrameとして取得
    print('-------------------------- End Scoring fn (get_dai_new_predictions) --------------------------')

    q.page['predictions'] = ui.form_card(
        box='4 2 8 5',
        items=[ui_table_from_df(df=df, name='Predictions Table', downloadable=True, height='400px')]
    )
    if q.client.modeling_task == 'regression':   # Regressionだと、データの表示のみ
        return

    grouped_predictions = df[df.columns[-1]].value_counts(bins=np.linspace(0, 1, 11), sort=False).reset_index()
    grouped_predictions['label'] = (grouped_predictions['index'].array.right * 100).astype(str) + '%'

    grouped_predictions = grouped_predictions.drop(['index'], axis=1)
    grouped_predictions.columns = ['count', 'label']

    q.page['distribution'] = ui.plot_card(
        box='4 7 8 3',
        title='Prediction Rates',
        data=data(
            fields=grouped_predictions.columns.tolist(),
            rows=grouped_predictions.values.tolist(),
            pack=True,
        ),
        plot=ui.plot(marks=[ui.mark(
            type='interval',
            x='=label', x_title='Prediction Interval',
            y='=count', y_title='Number of Observations',
            color='purple', stroke_color='black'
        )])
    )


##########################  ユーティリティ関数  ##########################

def ui_table_from_df(
    df: pd.DataFrame,
    name: str = 'table',
    sortables: list = None,
    filterables: list = None,
    searchables: list = None,
    min_widths: dict = None,
    max_widths: dict = None,
    multiple: bool = False,
    groupable: bool = False,
    downloadable: bool = False,
    link_col: str = None,
    height: str = '100%'
) -> ui.table:

    #print(df.head())

    if not sortables:
        sortables = []
    if not filterables:
        filterables = []
    if not searchables:
        searchables = []
    if not min_widths:
        min_widths = {}
    if not max_widths:
        max_widths = {}

    columns = [ui.table_column(
        name=str(x),
        label=str(x),
        sortable=True if x in sortables else False,
        filterable=True if x in filterables else False,
        searchable=True if x in searchables else False,
        min_width=min_widths[x] if x in min_widths.keys() else None,
        max_width=max_widths[x] if x in max_widths.keys() else None,
        link=True if x == link_col else False
    ) for x in df.columns.values]

    try:
        table = ui.table(
            name=name,
            columns=columns,
            rows=[
                ui.table_row(
                    name=str(i),
                    cells=[str(df[col].values[i]) for col in df.columns.values]
                ) for i in range(df.shape[0])
            ],
            multiple=multiple,
            groupable=groupable,
            downloadable=downloadable,
            height=height
        )
    except Exception:
        print(Exception)
        table = ui.table(
            name=name,
            columns=[ui.table_column('x', 'x')],
            rows=[ui.table_row(name='ndf', cells=[str('No data found')])]
        )

    return table
