"""
[2021/9/7]
予測結果データの表示に役立つPlotの例
- テーブル形式（0/1分類）
- 時系列データ

注：データの読み込みパス指定上（df_table, df_ts）、Wave_Tutorial_v0170ディレクトリからAppの起動を想定
"""

import os 
from h2o_wave import Q, main, app, ui, data, handle_on, on
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def prep_datasets_tabular(q: Q):
    ''' テーブル型データのデータ準備
    '''
    df_table = pd.read_csv(os.path.join('sample_data','titanic1_prediction_shapley.csv'))
    ## 予測値の分布用
    prediction_cut = pd.cut(df_table['survived.1'], bins=[i/10 for i in range(11)])   # [0,1]予測確率を10等分
    q.client.df_prediction_summary = pd.DataFrame(prediction_cut.value_counts()).reset_index().sort_values('index')
    ## Shapley
    contrib_cols = ['contrib_age','contrib_cabin','contrib_fare','contrib_parch','contrib_pclass','contrib_sex','contrib_sibsp','contrib_ticket']
    # Global Shapley用
    df_global_shapley = df_table[contrib_cols].apply(np.abs).apply(sum)
    df_global_shapley = pd.DataFrame(df_global_shapley).reset_index()
    df_global_shapley.columns = ['index','value']
    q.client.df_global_shapley = df_global_shapley.sort_values('value')
    ## 個別予測結果表示のID選択用
    id_list = df_table['Passenger_Id'].values.tolist()
    q.client.id_list = list(map(str, id_list))   # ID list with str data type
    ## 個別のShapley表示用
    df_all_shapley = df_table[['Passenger_Id', 'survived.1', *contrib_cols]].copy()
    df_all_shapley['Passenger_Id'] = df_all_shapley['Passenger_Id'].astype(str)
    q.client.df_all_shapley = df_all_shapley

def prep_datasets_ts(q: Q):
    ''' 時系列データのデータ準備
    '''
    df_ts = pd.read_csv(os.path.join('sample_data','walmart1_prediction.csv'))
    ## 合計データ
    df_ts_all = df_ts[['Date','Weekly_Sales','Weekly_Sales.predicted']].copy()
    q.client.df_ts_all = df_ts_all.groupby('Date').sum().reset_index()
    # 合計データ統計量
    q.client.n_len = len(q.client.df_ts_all['Date'])
    q.client.all_sum_actyal = q.client.df_ts_all['Weekly_Sales'].sum()
    q.client.all_sum_pred = q.client.df_ts_all['Weekly_Sales.predicted'].sum()
    q.client.all_rmse = np.sqrt(mean_squared_error(q.client.df_ts_all['Weekly_Sales'], q.client.df_ts_all['Weekly_Sales.predicted']))
    ## Groupデータ
    group_cols = ['Store', 'Dept']
    df_group_cols = df_ts[group_cols].copy()
    df_group_cols = df_group_cols.drop_duplicates()
    q.client.group_list = [(row[0], row[1]) for _,row in df_group_cols.iterrows()]
    # Store,Dept別で抽出して利用するデータ
    q.client.df_ts_individual = df_ts[['Store','Dept','Date','Weekly_Sales','Weekly_Sales.predicted']].copy()


@app('/plot')
async def serve(q: Q):
    print("<<<<<< App Start >>>>>>")

    if not q.client.initialized:
        # 初期表示Plot種類
        q.client.display_plot_type = 'tabular'    # one of 'tabular', 'ts'
        prep_datasets_tabular(q)
        prep_datasets_ts(q)
        q.client.initialized = True
        print('>> initialization done')
    
    q.page['header'] = ui.header_card(
        box='1 1 6 1',    # x座標 y座標 幅 高さ
        #box='header',
        title='Plotのサンプル',
        subtitle='データ形式に応じたPlotの例（左メニューから種類を選択）',
        nav=[
            ui.nav_group('Plotメニュー', items=[
                ui.nav_item(name='plot_type_tabular', label='テーブル形式（0/1分類）'),
                ui.nav_item(name='plot_type_ts', label='時系列'),
            ])
        ]
    )

    if q.args.plot_type_ts:
        q.client.display_plot_type = 'ts'
    elif q.args.plot_type_tabular:
        q.client.display_plot_type = 'tabular'
    else:
        pass
    
    if q.client.display_plot_type == 'ts':
        del q.page['tabular_info'], q.page['tabular_global_predict'], q.page['tabular_global_shapley'], q.page['tabular_local_select'], q.page['tabular_pred_proba'], q.page['tabular_local_shapley']
        render_ts_plots(q)
    elif q.client.display_plot_type == 'tabular':
        del q.page['ts_info'], q.page['ts_all_plot'], q.page['ts_all_stats'], q.page['ts_pick_ts']
        if q.client.individual_plot_initialized:   # 個別プロットの初期化
            for pn in q.client.page_names:
                del q.page[pn]
        render_tabular_plots(q)
    else:
        pass


    await handle_on(q)

    print("q.args --> ", q.args)
    #print("q.client --> ", q.client)
    print("q.client.display_plot_type --> ", q.client.display_plot_type)
    print("<<<<<< App End >>>>>>")
    await q.page.save()


####################################### テーブル形式（0/1分類） Start #######################################
def render_tabular_plots(q: Q):
    """ [tabular] Plot
    """
    q.page['tabular_info'] = ui.section_card(
        box='1 2 6 1',    # x座標 y座標 幅 高さ
        title='テーブル形式（0/1分類）のPlot例',
        subtitle='左側に、データ全体の予測結果分布と変数重要度（Global Shapley）。右側に、選択形式で各オブザベーションに対する予測結果と変数の寄与度（Shapley）。',
        items=[]
    )

    q.page['tabular_global_predict'] = ui.plot_card(
        box='1 3 3 3',
        title='全体的な予測値の分布',
        data=data(['index','pred'], rows=[(str(i),j) for i,j in zip(q.client.df_prediction_summary['index'], q.client.df_prediction_summary['survived.1'])]),
        plot=ui.plot([ui.mark(type='interval', x='=index', y='=pred')])
    )

    q.page['tabular_global_shapley'] = ui.plot_card(
        box='1 6 3 5',
        title='全体的な変数の重要度（Global Shapley）',
        data=data(['index','value'], rows=[(i,j) for i,j in zip(q.client.df_global_shapley['index'], q.client.df_global_shapley['value'])]),
        plot=ui.plot([ui.mark(type='interval', x='=value', y='=index',color='$blue')])
    )

    q.page['tabular_local_select'] = ui.form_card(
        box='4 3 3 1',
        items=[
            ui.dropdown(name='passenger_id', label='データの選択', choices=[ui.choice(i,i) for i in q.client.id_list], value='', trigger=True)
        ]
    )

@on('passenger_id')
async def show_individual_tabular_result(q: Q):
    """ [tabular] Plot
        DropdownからIDが選ばれた時の処理
    """
    q.page['card_local_select'].items[0].dropdown.value = q.args.passenger_id

    ## 個別表示用のデータの作成
    df_obs = q.client.df_all_shapley[q.client.df_all_shapley['Passenger_Id']==q.args.passenger_id].T.reset_index()
    df_obs.columns = ['index', 'value']
    df_shapley = df_obs[2:].copy()
    df_shapley = df_shapley.sort_values('value')    # 選択されたデータののShapley
    pred_proba = df_obs.loc[df_obs['index']=='survived.1', 'value'].values[0]   # 選択されたデータのの予測確率

    q.page['tabular_pred_proba'] = ui.wide_gauge_stat_card(
        box='4 4 3 2',
        title='選ばれたデータの予測確率',
        value='{:.2f}%'.format(pred_proba*100),
        aux_value='({})'.format(pred_proba),
        progress=pred_proba,
        plot_color='$red',
    )
    
    q.page['tabular_local_shapley'] = ui.plot_card(
        box='4 6 3 5',
        title='選ばれたデータにおける、変数の予測への貢献（Shapley）',
        data=data(['index','value'], rows=[(i,j) for i,j in zip(df_shapley['index'], df_shapley['value'])]),
        plot=ui.plot([ui.mark(type='interval', x='=value', y='=index',color='$blue')])
    )
####################################### テーブル形式（0/1分類） End #######################################


####################################### 時系列 Start #######################################
def render_ts_plots(q: Q):
    """ [ts] Plot
    """
    q.page['ts_info'] = ui.section_card(
        box='1 2 6 1',    # x座標 y座標 幅 高さ
        title='時系列データのPlot例',
        subtitle='全時系列の合計実績と予測。選択された個別時系列の実績と予測の表示（同時に3つまで表示可）。',
        items=[]
    )

    # Plot用へデータ加工（実測、予測の積み上げ形式）
    plotdata_date = q.client.df_ts_all['Date'].to_list() * 2
    plotdata_value = q.client.df_ts_all['Weekly_Sales'].to_list() + q.client.df_ts_all['Weekly_Sales.predicted'].to_list()
    plotdata_group = ['Actual' for _ in range(len(q.client.df_ts_all['Date']))] + ['Pred' for _ in range(len(q.client.df_ts_all['Date']))]

    q.page['ts_all_plot'] = ui.plot_card(
        box='1 3 5 3',
        title='全時系列合計',
        data=data(['date','value','group'], rows=[(i,j,k) for i,j,k in zip(plotdata_date,plotdata_value,plotdata_group)]),
        plot=ui.plot([ui.mark(type='line', x='=date', y='=value', color='=group')])
    )

    q.page['ts_all_stats'] = ui.form_card(
        box='6 3 1 3',
        items=[
            ui.text_s('N: {}'.format(q.client.n_len)),
            ui.text_s('実績合計: {:.2f}'.format(q.client.all_sum_actyal)),
            ui.text_s('予測合計: {:.2f}'.format(q.client.all_sum_pred)),
            ui.text_s('RMSE: {:.2f}'.format(q.client.all_rmse)),
        ]
    )

    q.page['ts_pick_ts'] = ui.form_card(box='1 6 4 1', items=[
            ui.picker(name='pickeed_ts', label='表示する個別時系列の選択（最大３）', 
                choices=[ui.choice(name=str(i), label='(Store,Dept):{}'.format(i)) for i in q.client.group_list],
                values=['spam', 'eggs'], 
                max_choices=3),
            ui.button(name='set_show_ts', label='選択', primary=True),
    ])

@on('set_show_ts')
async def show_individual_ts_result(q: Q):
    if q.client.individual_plot_initialized:   # 個別プロット作成が再選択された時の初期化
        for pn in q.client.page_names:
            del q.page[pn]
    
    selected_ts_id = [tuple(map(int, i.strip('()').split(','))) for i in q.args.pickeed_ts]   # 文字列をタプルに変換

    q.client.page_names = []   # 個別時系列（plotと統計量）のページ名保存用
    for counter,ts_id in enumerate(selected_ts_id):    # 個別時系列のPlotと統計量

        y_loc = 7 + 3*counter   # y軸の開始位置

        ## 対象のデータ抽出
        df_one_ts = q.client.df_ts_individual.loc[(q.client.df_ts_individual['Store']==ts_id[0])&(q.client.df_ts_individual['Dept']==ts_id[1]), ['Date','Weekly_Sales','Weekly_Sales.predicted']].copy()
        # Plot用へデータ加工（実測、予測の積み上げ形式）
        plotdata_date = df_one_ts['Date'].to_list() * 2
        plotdata_value = df_one_ts['Weekly_Sales'].to_list() + df_one_ts['Weekly_Sales.predicted'].to_list()
        plotdata_group = ['Actual' for _ in range(len(df_one_ts['Date']))] + ['Pred' for _ in range(len(df_one_ts['Date']))]
        # 統計量
        individual_sum_actual = df_one_ts['Weekly_Sales'].sum()
        individual_sum_pred = df_one_ts['Weekly_Sales.predicted'].sum()
        individual_rmse = np.sqrt(mean_squared_error(df_one_ts['Weekly_Sales'], df_one_ts['Weekly_Sales.predicted']))

        plot_page_name = 'ts_individual_plot_{}'.format(counter)
        q.client.page_names.append(plot_page_name)
        q.page[plot_page_name] = ui.plot_card(
            box='1 {} 5 3'.format(y_loc),
            title='個別時系列：{}'.format(ts_id),
            data=data(['date','value','group'], rows=[(i,j,k) for i,j,k in zip(plotdata_date,plotdata_value,plotdata_group)]),
            plot=ui.plot([ui.mark(type='line', x='=date', y='=value', color='=group')])
        )

        stats_page_name = 'ts_individual_stats_{}'.format(counter)
        q.client.page_names.append(stats_page_name)
        q.page[stats_page_name] = ui.form_card(
            box='6 {} 1 3'.format(y_loc),
            items=[
                ui.text_m('ID:{}'.format(ts_id)),
                ui.text_s('実績合計: {:.2f}'.format(individual_sum_actual)),
                ui.text_s('予測合計: {:.2f}'.format(individual_sum_pred)),
                ui.text_s('RMSE: {:.2f}'.format(individual_rmse)),
            ]
    )
    q.client.individual_plot_initialized = True

####################################### 時系列 End #######################################

