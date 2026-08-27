from pathlib import Path

import duckdb
import pandas as pd


DATA_DIR = Path("data")
CSV_FILES = sorted(DATA_DIR.glob("historico_*.csv"))


def main():
    if len(CSV_FILES) < 5:
        raise RuntimeError(f"Se esperaban al menos 5 historicos CSV; encontrados: {len(CSV_FILES)}")

    total = 0
    for csv_path in CSV_FILES:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise RuntimeError(f"Historico vacio: {csv_path}")
        pq_path = csv_path.with_suffix(".parquet")
        # ZSTD ofrece buena compresion y lectura selectiva eficiente.
        df.to_parquet(pq_path, index=False, engine="pyarrow", compression="zstd")
        con = duckdb.connect(database=":memory:")
        try:
            pq_n = int(con.execute("SELECT count(*) FROM read_parquet(?)", [str(pq_path)]).fetchone()[0])
        finally:
            con.close()
        if pq_n != len(df):
            raise RuntimeError(f"Paridad CSV/Parquet invalida: {csv_path}={len(df)} vs {pq_path}={pq_n}")
        total += pq_n
        print(f"PARQUET_OK {pq_path}: {pq_n} filas")

    print(f"PARQUET_STORE_OK total={total} files={len(CSV_FILES)}")


if __name__ == "__main__":
    main()
