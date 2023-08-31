import streamlit as st
import numpy as np
import pandas as pd
import datetime as datetime
import pandas_datareader
import plotly.graph_objects as go

import sklearn.linear_model
import sklearn.model_selection
from PIL import Image
import yfinance as yf
yf.pdr_override()

st.title('AIで株価予測アプリ')
st.write('予測させてみましょう。')

#トップ画像の表示
image = Image.open('stock_predict.png')
st.image(image, use_column_width=True)

st.write("※あくまでもAIによる予測値ですので、あしからず。")
st.header("株価銘柄のティッカーシンボルを入力してください。")
stock_name = st.text_input("例：AAPL, FB, SFTBY (大文字小文字どちらでも可)", "AAPL")
stock_name = stock_name.upper()

link= 'https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_usequity_list.html'
st.markdown(link)
st.write('ティッカーシンボルについては上のリンク(SBI証券)をご覧ください。')

df_stock = pandas_datareader.data.get_data_yahoo(stock_name, '2021-01-05')
st.header(stock_name + "2022年１月５日から現在までの価格(USD")
st.write(df_stock)

st.header(stock_name + "終値と１４日間の平均(USD)")
df_stock['SMA'] = df_stock['Close'].rolling(window=14).mean()
df_stock2 = df_stock[['Close', 'SMA']]
st.line_chart(df_stock2)

st.header(stock_name + "値動き(USD)")
df_stock['change'] = (((df_stock['Close'] - df_stock['Open'])) / (df_stock['Open']) * 100)
st.line_chart(df_stock['change'].tail(100))

fig = go.Figure(
    data = [go.Candlestick(
        x = df_stock.index,
        open = df_stock['Open'],
        high = df_stock['High'],
        low = df_stock['Low'],
        close = df_stock['Close'],
        increasing_line_color = 'green',
        decreasing_line_color = 'red',
    )
    ]
)
st.header(stock_name + "キャンドルスティック")
st.plotly_chart(fig, use_container_width=True)

df_stock['label'] = df_stock['Close'].shift(-30)

st.header(stock_name + '１か月後の予測(USD)')
def stock_presict():
    X = np.array(df_stock.drop(['label', 'SMA'], axis=1))
    # 値が大きいので、平均を引いて、標準偏差で割って、スケーリングする
    X = sklearn.preprocessing.scale(X)
    # 過去30日間のデータ
    predict_data = X[-30:]
    # 過去30日間を取り除いた入力データ
    X = X[:-30]
    y = np.array(df_stock['label'])
    # 過去３０日を取り除いた正解ラベル
    y = y[:-30]
    # 訓練データ８０％、検証データ２０％
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
    # 訓練データをもとに学習
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)
    # 検証
    accuracy = model.score(X_test, y_test)
    st.write(f'正答率は{round((accuracy) * 100, 1)}%です。')

    #accuracyより信頼性を表示
    if accuracy > 0.75:
        st.write('信頼度:高')
    elif accuracy >0.5:
        st.write('信頼度:中')
    else:
        st.write('信頼度:低')
    st.write('オレンジ線が予測値です。')

    #検証
    predicted_data = model.predict(predict_data)
    df_stock['Predict'] = np.nan
    last_date = df_stock.iloc[-1].name
    one_day = 86400
    next_unix = last_date.timestamp() + one_day

    #予測をグラフ化
    for data in predicted_data:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += one_day
        #     print(next_date)
        # np.appendでnp.nanを追加、
        df_stock.loc[next_date] = np.append([np.nan] * (len(df_stock.columns) - 1), data)

    df_stock['Close'].plot(figsize=(15,6), color="green")
    df_stock['Predict'].plot(figsize=(15,6), color="red")

    df_stock3 = df_stock[['Close', 'Predict']]
    st.line_chart(df_stock3)

#ボタンで予測開始
if st.button('予測する'):
    stock_presict()