# Fill-price realism check: NOT RUN YET (no real numbers)

The scripts are written, but their network stages are UNTESTED. The Polymarket APIs (gamma and data-api) are
blocked on the machine where the scripts were written, and the block was not bypassed.

To run them in a cloud environment, start from the repo root:

    python src/fill_check.py sample    # offline and deterministic (seed 0): 1500 favourites + 500 non-favourites
    python src/fill_check.py verify    # 3 markets: prints Gamma/trade fields to confirm side/asset/outcomeIndex semantics
    python src/fill_check.py fetch     # Gamma conditionId/clobTokenIds + takerOnly trades, cached in data/raw/fill_cache/
    python src/fill_check.py analyze   # writes results/fill_check.json and fill_check_markets.csv.gz, and overwrites this file
    python src/stations.py             # writes results/stations.json

Only `sample` has run on real data. `analyze` was smoke-tested on synthetic trades built with a known
+-2c spread. It recovered ask-p = +0.020 and bid-p = -0.020, and after mapping the NO-token and YES-token
prices agreed.
