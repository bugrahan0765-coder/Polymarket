"""Second pass over the HF on-chain fills: per (wallet, market) position aggregates, for the smart-money study.
Each fill row has a maker and a taker; both get a record (role, direction on the traded token).
Output <out>/wallets/rg*.parquet with: wallet, market_id, role, tok (1/2), buy (bool), shares, usd, n, t_first, t_last.
Resumable; each row group in its own process. Run: python src/hf_extract_wallets.py <out_dir> [threads] [reverse]
"""
import os, sys, time, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import fsspec, pyarrow.parquet as pq, pandas as pd

URL = 'https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data/resolve/main/trades.parquet'
COLS = ['timestamp', 'market_id', 'maker', 'taker', 'token_amount', 'usd_amount', 'maker_direction', 'nonusdc_side']
FIRST_RG = 104
OUT = sys.argv[1]; THREADS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != '--rg' else 6
os.makedirs(os.path.join(OUT, 'wallets'), exist_ok=True)


def work(i):
    done = os.path.join(OUT, 'wallets', f'rg{i:04d}.parquet')
    if os.path.exists(done): return i, 0
    for a in range(6):
        try:
            pf = pq.ParquetFile(fsspec.filesystem('https').open(URL, block_size=8 * 2**20))
            t = pf.read_row_group(i, columns=COLS).to_pandas(); break
        except Exception as e:
            time.sleep(10 * (a + 1)); err = e
    else:
        raise RuntimeError(f'rg {i}: {err}')
    t['timestamp'] = t['timestamp'].astype('int64')
    t['tok'] = (t['nonusdc_side'] == 'token1').map({True: 1, False: 2}).astype('int8')
    mk = t[['timestamp', 'market_id', 'maker', 'tok', 'token_amount', 'usd_amount']].rename(columns={'maker': 'wallet'})
    mk['buy'] = t['maker_direction'] == 'BUY'; mk['role'] = 'M'
    tk = t[['timestamp', 'market_id', 'taker', 'tok', 'token_amount', 'usd_amount']].rename(columns={'taker': 'wallet'})
    tk['buy'] = t['maker_direction'] != 'BUY'; tk['role'] = 'T'
    x = pd.concat([mk, tk], ignore_index=True)
    g = x.groupby(['wallet', 'market_id', 'role', 'tok', 'buy']).agg(shares=('token_amount', 'sum'), usd=('usd_amount', 'sum'), n=('usd_amount', 'size'),
                                                                     t_first=('timestamp', 'min'), t_last=('timestamp', 'max')).reset_index()
    g.to_parquet(done + '.tmp', index=False, compression='zstd'); os.replace(done + '.tmp', done)
    return i, len(g)


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
