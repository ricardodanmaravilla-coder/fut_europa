from pathlib import Path

import duckdb
import pandas as pd

from modules.data_store import cargar_historico
from modules.montecarlo_europa import MC_PROFILE_PARQUET, build_mc_profile_rows


DATA_DIR = Path("data")
CSV_FILES = sorted(DATA_DIR.glob("historico_*.csv"))
LEAGUE_BY_FILE = {
    "historico_premier.csv": "Premier League",
    "historico_laliga.csv": "La Liga",
    "historico_seriea.csv": "Serie A",
    "historico_bundesliga.csv": "Bundesliga",
    "historico_ligue1.csv": "Ligue 1",
}


def main():
    if len(CSV_FILES) < 5:
        raise RuntimeError(f"Se esperaban al menos 5 historicos CSV; encontrados: {len(CSV_FILES)}")

    total = 0
    for csv_path in CSV_FILES:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise RuntimeError(f"Historico vacio: {csv_path}")
        pq_path = csv_path.with_suffix(".parquet")
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

    mc_rows = []
    for csv_path in CSV_FILES:
        league = LEAGUE_BY_FILE.get(csv_path.name)
        if not league:
            continue
        df = cargar_historico(str(csv_path))
        mc_rows.extend(build_mc_profile_rows(df, league))
    if not mc_rows:
        raise RuntimeError("No se pudo construir el cache de perfiles Monte Carlo")
    mc_df = pd.DataFrame(mc_rows)
    MC_PROFILE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    mc_df.to_parquet(MC_PROFILE_PARQUET, index=False, engine="pyarrow", compression="zstd")

    # Paridad fuerte: cada perfil precalculado debe coincidir con el calculado desde el histórico.
    for csv_path in CSV_FILES:
        league = LEAGUE_BY_FILE.get(csv_path.name)
        if not league:
            continue
        df = cargar_historico(str(csv_path))
        expected = pd.DataFrame(build_mc_profile_rows(df, league)).sort_values(["team", "venue"]).reset_index(drop=True)
        got = mc_df[mc_df["league"] == league].sort_values(["team", "venue"]).reset_index(drop=True)
        if len(expected) != len(got):
            raise RuntimeError(f"MC cache incompleto para {league}: {len(got)} != {len(expected)}")
        for col in ("xgf", "xga", "cf", "ca", "cards"):
            diff = (expected[col] - got[col]).abs().max()
            if float(diff or 0.0) > 1e-12:
                raise RuntimeError(f"MC cache sin paridad en {league}/{col}: diff={diff}")

    print(f"MC_PROFILE_PARQUET_OK {MC_PROFILE_PARQUET}: {len(mc_df)} perfiles")
    print(f"PARQUET_STORE_OK total={total} files={len(CSV_FILES)}")


if __name__ == "__main__":
    main()
