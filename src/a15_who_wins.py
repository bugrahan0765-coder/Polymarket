"""Step 15: who makes money on Polymarket, and how? Uses wallet x market PnL (wallet_pm.parquet from a8).
Per wallet: PnL before SEL_END and after TEST_START, maker share of volume, category mix, style
(directional = held net position to resolution vs flat = bought and sold/merged, |net| small vs volume).
Persistent winners = PnL > 0 in both periods with >= 20 markets each. Reports where their PnL comes from
(category x maker/taker x style) and the same for all wallets.
Run: python src/a15_who_wins.py <hf_data_dir>
"""
import sys, json, duckdb, pandas as pd

HF = sys.argv[1] if len(sys.argv) > 1 else '../hf_data'
SEL_END = pd.Timestamp('2026-04-01', tz='UTC').timestamp()
TEST_START = pd.Timestamp('2026-05-01', tz='UTC').timestamp()
c = duckdb.connect(); c.execute("SET memory_limit='4GB'"); c.execute('SET threads=3'); c.execute("SET temp_directory='../hf_data/duck_tmp'")
c.execute('SET preserve_insertion_order=false')
c.execute(f"""create view pm as select *, case when end_ts < {SEL_END} then 'A' when end_ts >= {TEST_START} then 'B' else 'X' end per,
  case when cat = 'crypto' and tb1 is not null and end_ts - least(coalesce(tb1, 9e18), coalesce(tb2, 9e18)) < 7200 then 'crypto_short' else cat end cat2,
  case when abs(net1) + abs(net2) < 0.2 * greatest(bought, 1) then 'flat' else 'directional' end style
  from read_parquet('{HF}/wallet_pm.parquet') where bought > 0""")
c.execute("""create table infra as select wallet from pm group by wallet having count(*) > 200000""")
c.execute("""create table w as select wallet,
  sum(case when per='A' then pnl end) pnl_a, sum(case when per='B' then pnl end) pnl_b,
  sum((per='A')::int) n_a, sum((per='B')::int) n_b, sum(bought) vol, sum(pnl) pnl, avg(taker_share) taker
  from pm where wallet not in (select wallet from infra) group by wallet""")
print(c.execute("""select count(*) wallets, sum((pnl > 0)::int) winners, sum(pnl) total_pnl, sum(case when pnl>0 then pnl end) winners_pnl,
                   sum(case when pnl<0 then pnl end) losers_pnl from w""").df().to_string())
c.execute("create table pw as select wallet from w where n_a >= 20 and n_b >= 20 and pnl_a > 0 and pnl_b > 0")
c.execute("create table both_ as select wallet from w where n_a >= 20 and n_b >= 20")
print(c.execute("select (select count(*) from pw) as n_persistent, (select count(*) from both_) as active_both,"
                " (select count(*) from w where n_a>=20 and n_b>=20 and pnl_a>0) as won_a").df().to_string())
# persistence by taker share bucket
print(c.execute("""select round(taker, 1) taker_bucket, count(*) n, avg((pnl_b > 0)::int) p_win_b_given_a,
  sum(pnl_b) pnl_b from w where n_a >= 20 and n_b >= 20 and pnl_a > 0 group by 1 order by 1""").df().to_string())
for name, filt in [('persistent winners', 'wallet in (select wallet from pw)'), ('everyone', 'wallet not in (select wallet from infra)')]:
    d = c.execute(f"""select cat2, style, case when taker_share >= .5 then 'taker' else 'maker' end as mrole,
      count(*) as positions, sum(bought) as vol, sum(pnl) as pnl, sum(pnl) / sum(bought) as roi from pm where per = 'B' and {filt} group by 1, 2, 3 order by pnl desc""").df()
    print(f'\n=== test period PnL by category x style x role: {name} ===\n' + d.round(4).to_string(index=False))
    d.to_csv(f"results/a15_{name.replace(' ', '_')}.csv", index=False)
top = c.execute("""select w.wallet, n_a, n_b, round(pnl_a) pnl_a, round(pnl_b) pnl_b, round(vol) vol, round(taker, 2) taker from w
  join pw using (wallet) order by pnl_b desc limit 30""").df()
print('\n=== top persistent winners ===\n' + top.to_string(index=False))
# what the top 30 trade
ids = "','".join(top.wallet)
print(c.execute(f"""select wallet, cat2, style, count(*) n, round(sum(pnl)) pnl, round(sum(pnl)/sum(bought), 4) roi, round(avg(taker_share), 2) taker
  from pm where per = 'B' and wallet in ('{ids}') group by 1, 2, 3 qualify row_number() over (partition by wallet order by sum(pnl) desc) <= 2
  order by wallet, pnl desc""").df().to_string(index=False))
