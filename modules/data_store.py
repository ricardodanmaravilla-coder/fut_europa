import os
from pathlib import Path

import duckdb
import pandas as pd


DATA_DIR = Path("data")


def parquet_path(csv_path: str) -> str:
    return str(Path(csv_path).with_suffix(".parquet"))


def cargar_historico(csv_path: str) -> pd.DataFrame:
    """Carga preferentemente Parquet mediante DuckDB; CSV queda como fallback seguro."""
    pq = parquet_path(csv_path)
    if os.path.exists(pq):
        try:
            con = duckdb.connect(database=":memory:")
            try:
                df = con.execute(
                    "SELECT * FROM read_parquet(?) ORDER BY Fecha, Local, Visitante",
                    [pq],
                ).fetch_df()
            finally:
                con.close()
            return df
        except Exception:
            pass
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def consultar_historico(csv_path: str, fecha_desde=None, fecha_hasta=None, equipos=None) -> pd.DataFrame:
    """Consulta selectiva con DuckDB sin cargar todo el histórico en RAM."""
    pq = parquet_path(csv_path)
    if not os.path.exists(pq):
        df = cargar_historico(csv_path)
        if df.empty:
            return df
        if fecha_desde is not None:
            df = df[df["Fecha"].astype(str) >= str(fecha_desde)]
        if fecha_hasta is not None:
            df = df[df["Fecha"].astype(str) <= str(fecha_hasta)]
        if equipos:
            eq = set(map(str, equipos))
            df = df[df["Local"].isin(eq) | df["Visitante"].isin(eq)]
        return df.reset_index(drop=True)

    clauses = []
    params = [pq]
    if fecha_desde is not None:
        clauses.append("Fecha >= ?")
        params.append(str(fecha_desde))
    if fecha_hasta is not None:
        clauses.append("Fecha <= ?")
        params.append(str(fecha_hasta))
    if equipos:
        equipos = list(map(str, equipos))
        marks = ",".join(["?"] * len(equipos))
        clauses.append(f"(Local IN ({marks}) OR Visitante IN ({marks}))")
        params.extend(equipos)
        params.extend(equipos)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f"SELECT * FROM read_parquet(?) {where} ORDER BY Fecha, Local, Visitante"
    con = duckdb.connect(database=":memory:")
    try:
        return con.execute(sql, params).fetch_df()
    finally:
        con.close()


def validar_paridad(csv_path: str) -> dict:
    """Comprueba que CSV y Parquet representan el mismo dataset lógico."""
    pq = parquet_path(csv_path)
    if not (os.path.exists(csv_path) and os.path.exists(pq)):
        return {"ok": False, "csv": os.path.exists(csv_path), "parquet": os.path.exists(pq)}
    csv_n = len(pd.read_csv(csv_path, usecols=["Fecha"]))
    con = duckdb.connect(database=":memory:")
    try:
        pq_n = int(con.execute("SELECT count(*) FROM read_parquet(?)", [pq]).fetchone()[0])
    finally:
        con.close()
    return {"ok": csv_n == pq_n, "csv_rows": csv_n, "parquet_rows": pq_n}
