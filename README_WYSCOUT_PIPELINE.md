# Wyscout representation layer

This optional research layer uses the public Wyscout 2017/18 top-five European league event dataset (CC BY 4.0) through socceraction. It converts provider events to SPADL, learns an xThreat grid, computes PPDA research features and evaluates VAEP only on a held-out tail of games. It is deliberately kept separate from the 2021+ prediction history so old matches cannot contaminate the production walk-forward.

Outputs are Parquet files under `data/` and are intended to pretrain or audit action representations. They must not receive production weight until a separate temporal validation demonstrates improvement on modern data.
