"""Extract Polymarket on-chain fills (HF dataset SII-WANGZJ/Polymarket_data, trades.parquet) without downloading it whole.
One pass over row groups since Oct 2025, reading only the needed columns over HTTP:
  - <out>/weather_trades/rg*.parquet : every fill of the markets in data/decision_prices.csv.gz
  - <out>/hourly/rg*.parquet         : all markets, per (market_id, hour): token1 taker-buy / taker-sell VWAP, counts, USD volume
Prices are expressed for token1 (answer1, usually "Yes"): a taker BUY of token1 at x, or taker SELL of token2 at x,
both mean token1 was bought at the ask (x or 1-x). Resumable: finished row groups are skipped.
Run: python src/hf_extract.py <out_dir> [threads] [reverse]
"""
import os, sys, time, gzip, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import fsspec, pyarrow.parquet as pq, pandas as pd, numpy as np

URL = 'https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data/resolve/main/trades.parquet'
COLS = ['timestamp', 'market_id', 'price', 'usd_amount', 'taker_direction', 'nonusdc_side']
FIRST_RG = 104  # first row group reaching 2025-10-01
OUT = sys.argv[1]; THREADS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != '--rg' else 6
WEATHER = {r['market_id'] for r in csv.DictReader(gzip.open('data/decision_prices.csv.gz', 'rt'))}
for d in ['weather_trades', 'hourly']: os.makedirs(os.path.join(OUT, d), exist_ok=True)


def normalise(t):
    tok1 = t['nonusdc_side'] == 'token1'
    t['p1'] = np.where(tok1, t['price'], 1 - t['price'])
    t['buy1'] = (t['taker_direction'] == 'BUY') == tok1
    return t


def work(i):
    done = os.path.join(OUT, 'hourly', f'rg{i:04d}.parquet')
    if os.path.exists(done): return i, 0, 0
    if os.path.exists(done + '.tmp'): return i, -1, -1  # another runner is on it
    for a in range(6):
        try:
            pf = pq.ParquetFile(fsspec.filesystem('https').open(URL, block_size=8 * 2**20))
            t = pf.read_row_group(i, columns=COLS).to_pandas()
            break
        except Exception as e:
            time.sleep(10 * (a + 1)); err = e
    else:
        raise RuntimeError(f'rg {i}: {err}')
    t['timestamp'] = t['timestamp'].astype('int64'); t = normalise(t)
    w = t[t['market_id'].isin(WEATHER)]
    w.to_parquet(os.path.join(OUT, 'weather_trades', f'rg{i:04d}.parquet'), index=False, compression='zstd')
    t['hour'] = t['timestamp'] // 3600 * 3600
    t['pv'] = t['p1'] * t['usd_amount']
    g = t.groupby(['market_id', 'hour', 'buy1']).agg(n=('p1', 'size'), usd=('usd_amount', 'sum'), pv=('pv', 'sum'),
                                                     pmin=('p1', 'min'), pmax=('p1', 'max'), plast=('p1', 'last')).reset_index()
    g['vwap'] = g['pv'] / g['usd'].where(g['usd'] > 0)
    g.drop(columns='pv').to_parquet(done + '.tmp', index=False, compression='zstd'); os.replace(done + '.tmp', done)
    return i, len(t), len(w)


def run_isolated(i):
    """Each row group in its own process, so one crashed worker cannot take down the run."""
    import subprocess
    for a in range(5):
        r = subprocess.run([sys.executable, __file__, OUT, '--rg', str(i)], capture_output=True, text=True)
        if r.returncode == 0: return i, r.stdout.strip()
        time.sleep(2)
    return i, f'FAILED rc={r.returncode} {r.stderr[-300:]}'


if __name__ == '__main__':
    if '--rg' in sys.argv:
        print(work(int(sys.argv[sys.argv.index('--rg') + 1]))); sys.exit(0)
    n = pq.ParquetFile(fsspec.filesystem('https').open(URL)).metadata.num_row_groups
    t0 = time.time(); k = 0
    with ThreadPoolExecutor(THREADS) as ex:
        order = range(FIRST_RG, n)
        if len(sys.argv) > 3 and sys.argv[3] == 'reverse': order = reversed(order)  # second runner works from the end
        for fut in as_completed([ex.submit(run_isolated, i) for i in order]):
            i, msg = fut.result(); k += 1
            if 'FAILED' in msg or k % 25 == 0: print(f'{k}/{n - FIRST_RG} {time.time() - t0:.0f}s rg {i}: {msg}', flush=True)
    print('done', time.time() - t0)
