"""Step 13a: Binance spot 1m klines (public bulk files, data.binance.vision) for the coins with Polymarket up/down markets.
Output ../hf_data/binance/<SYM>-1m.parquet (open_ts s, open, close). Run: python src/a13_binance_fetch.py
"""
import io, os, zipfile, urllib.request
import pandas as pd

SYMS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'BNBUSDT']
MONTHS = pd.period_range('2025-09', '2026-09', freq='M')
DAYS = pd.date_range('2026-10-01', '2026-10-05')
OUT = '../hf_data/binance'


def get(url):
    try:
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=120).read()))
    except Exception as e:
        print('  miss', url, e); return None
    d = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 1, 4], names=['t', 'open', 'close'])
    d['t'] = d['t'] // 1000 if d['t'].iloc[0] < 1e15 else d['t'] // 1_000_000  # ms before 2025, us after
    return d


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    base = 'https://data.binance.vision/data/spot'
    for s in SYMS:
        p = f'{OUT}/{s}-1m.parquet'
        if os.path.exists(p): continue
        parts = [get(f'{base}/monthly/klines/{s}/1m/{s}-1m-{m}.zip') for m in MONTHS]
        parts += [get(f'{base}/daily/klines/{s}/1m/{s}-1m-{d:%Y-%m-%d}.zip') for d in DAYS]
        d = pd.concat([x for x in parts if x is not None]).drop_duplicates('t').sort_values('t')
        d.to_parquet(p, compression='zstd', index=False)
        print(s, len(d), pd.to_datetime(d['t'].min(), unit='s'), pd.to_datetime(d['t'].max(), unit='s'), flush=True)
