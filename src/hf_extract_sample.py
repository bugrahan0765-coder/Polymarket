"""Third pass: every fill of a deterministic 5% sample of ALL markets (int(market_id) % 20 == 0), for point-in-time
(look-ahead-free) strategy tests outside weather. Same token1 normalisation as hf_extract.py.
Output <out>/sample_trades/rg*.parquet. Resumable, one process per row group.
Run: python src/hf_extract_sample.py <out_dir> [threads] [reverse]
"""
import os, sys, time, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import fsspec, pyarrow.parquet as pq, numpy as np, pandas as pd

URL = 'https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data/resolve/main/trades.parquet'
COLS = ['timestamp', 'market_id', 'price', 'usd_amount', 'taker_direction', 'nonusdc_side']
FIRST_RG, MOD = 104, 20
OUT = sys.argv[1]; THREADS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != '--rg' else 6
os.makedirs(os.path.join(OUT, 'sample_trades'), exist_ok=True)


def work(i):
    done = os.path.join(OUT, 'sample_trades', f'rg{i:04d}.parquet')
    if os.path.exists(done): return i, 0
    for a in range(6):
        try:
            pf = pq.ParquetFile(fsspec.filesystem('https').open(URL, block_size=8 * 2**20))
            t = pf.read_row_group(i, columns=COLS).to_pandas(); break
        except Exception as e:
            time.sleep(10 * (a + 1)); err = e
    else:
        raise RuntimeError(f'rg {i}: {err}')
    t = t[pd.to_numeric(t['market_id'], errors='coerce').fillna(1) % MOD == 0].copy()
    t['timestamp'] = t['timestamp'].astype('int64')
    tok1 = t['nonusdc_side'] == 'token1'
    t['p1'] = np.where(tok1, t['price'], 1 - t['price'])
    t['buy1'] = (t['taker_direction'] == 'BUY') == tok1
    t[['timestamp', 'market_id', 'p1', 'buy1', 'usd_amount']].to_parquet(done + '.tmp', index=False, compression='zstd')
    os.replace(done + '.tmp', done)
    return i, len(t)


def run_isolated(i):
    for a in range(5):
        r = subprocess.run([sys.executable, __file__, OUT, '--rg', str(i)], capture_output=True, text=True)
        if r.returncode == 0: return i, r.stdout.strip()
        time.sleep(2)
    return i, f'FAILED rc={r.returncode} {r.stderr[-300:]}'


if __name__ == '__main__':
    if '--rg' in sys.argv:
        print(work(int(sys.argv[sys.argv.index('--rg') + 1]))); sys.exit(0)
    n = pq.ParquetFile(fsspec.filesystem('https').open(URL)).metadata.num_row_groups
    order = range(FIRST_RG, n)
    if len(sys.argv) > 3 and sys.argv[3] == 'reverse': order = reversed(order)
    t0 = time.time(); k = 0
    with ThreadPoolExecutor(THREADS) as ex:
        for fut in as_completed([ex.submit(run_isolated, i) for i in order]):
            i, msg = fut.result(); k += 1
            if 'FAILED' in msg or k % 25 == 0: print(f'{k}/{n - FIRST_RG} {time.time() - t0:.0f}s rg {i}: {msg}', flush=True)
    print('done', time.time() - t0)
