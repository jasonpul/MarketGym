import datetime
import pandas as pd

epsilon = 1e-16


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = pd.Series.ewm(up, span=n).mean()
    ema_down = pd.Series.ewm(down, span=n).mean() + epsilon
    rsi = ema_up / ema_down
    rsi = 100 - (100 / (1 + rsi))

    return rsi + epsilon


def macd(series: pd.Series) -> pd.Series:
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    return macd + epsilon


def cci(close: pd.Series, high: pd.Series, low: pd.Series, n: int = 20) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    moving_average = typical_price.rolling(n).mean()
    mean_deviation = typical_price.rolling(n).std()
    cci = (typical_price - moving_average) / (0.15 * mean_deviation + epsilon)
    return cci + epsilon


# def get_adx(close: pd.Series, high: pd.Series, low: pd.Series, n: int = 14) -> pd.Series:
#     plus_dm = high.diff().clip(lower=0)
#     minus_dm = low.diff().clip(upper=0)

#     tr1 = pd.DataFrame(high - low)
#     tr2 = pd.DataFrame(abs(high - close.shift(1)))
#     tr3 = pd.DataFrame(abs(low - close.shift(1)))
#     frames = [tr1, tr2, tr3]
#     tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
#     atr = tr.rolling(n).mean()

#     plus_di = 100 * (plus_dm.ewm(alpha=1 / n).mean() / atr)
#     minus_di = abs(100 * (minus_dm.ewm(alpha=1 / n).mean() / atr))
#     dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
#     adx = ((dx.shift(1) * (n - 1)) + dx) / n
#     adx_smooth = adx.ewm(alpha=1 / n).mean()
#     return plus_di, minus_di, adx_smooth


def get_sp500_filtered_symbols(date: datetime.datetime = datetime.datetime(year=2008, month=1, day=1), n: int = 50):
    def clean_date(x):
        if type(x) is str:
            x = x.split(' ')[0].strip()
        else:
            x = '2021-01-01'
        return x
    payload = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500 = payload[0]
    sp500 = sp500.rename(columns={'Date first added': 'date_added'})
    sp500 = sp500[['Symbol', 'date_added', 'Founded']]

    payload = pd.read_html(
        'file:///home/hotdog/Downloads/sp500_market_cap.html')
    market_cap = payload[0].sort_values('Weight', ascending=False)
    market_cap = market_cap[['Symbol', 'Weight']]

    sp500 = sp500.merge(market_cap)
    sp500['date_added'] = sp500['date_added'].apply(clean_date)
    sp500['date_added'] = pd.to_datetime(sp500['date_added'])

    filtered = sp500[sp500['date_added'] <=
                     date].sort_values('Weight', ascending=False)

    return filtered.iloc[:n]['Symbol'].tolist()
