import io
import os
import unicodedata

import numpy as np
import pandas as pd
import requests

LIGAS = {
    "Premier League": {"codigo": "E0", "archivo": "data/historico_premier.csv"},
    "La Liga": {"codigo": "SP1", "archivo": "data/historico_laliga.csv"},
    "Serie A": {"codigo": "I1", "archivo": "data/historico_seriea.csv"},
    "Bundesliga": {"codigo": "D1", "archivo": "data/historico_bundesliga.csv"},
    "Ligue 1": {"codigo": "F1", "archivo": "data/historico_ligue1.csv"},
}

SEASON_CODES = ["2021", "2122", "2223", "2324", "2425", "2526", "2627"]
BASE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
STATSBOMB_CACHE = "data/statsbomb_xg_matches.csv"


def normalizar_nombre(nombre):
    txt = str(nombre or "").strip()
    return unicodedata.normalize("NFKD", txt).encode("ASCII", "ignore").decode("utf-8").strip()


def key_nombre(nombre):
    return "".join(ch.lower() for ch in normalizar_nombre(nombre) if ch.isalnum())


def n(row, key, default=np.nan):
    val = pd.to_numeric(row.get(key), errors="coerce")
    return default if pd.isna(val) else float(val)


def first_numeric(row, keys, default=np.nan):
    for key in keys:
        val = n(row, key, np.nan)
        if not pd.isna(val) and val > 1.0:
            return val
    return default


def no_vig_1x2(h, d, a):
    vals = np.array([h, d, a], dtype=float)
    if np.any(~np.isfinite(vals)) or np.any(vals <= 1.0):
        return (np.nan, np.nan, np.nan)
    q = 1.0 / vals
    q = q / q.sum()
    return tuple(float(x) for x in q)


def xg_proxy(shots, shots_on_target, goals):
    """Proxy transparente cuando la fuente no publica xG real."""
    s = 0.0 if pd.isna(shots) else float(shots)
    sot = 0.0 if pd.isna(shots_on_target) else float(shots_on_target)
    g = 0.0 if pd.isna(goals) else float(goals)
    return round(float(np.clip(0.055 * s + 0.18 * sot + 0.10 * g, 0.20, 4.00)), 2)


def descargar_csv(season_code, league_code):
    url = BASE.format(season=season_code, league=league_code)
    r = requests.get(url, timeout=20, headers={"User-Agent": "fut-europa/3.0"})
    if r.status_code != 200 or len(r.content) < 100:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.BytesIO(r.content))
    except Exception:
        return pd.DataFrame()


def transformar(df):
    if df is None or df.empty:
        return pd.DataFrame()
    required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        local = normalizar_nombre(row.get("HomeTeam"))
        visita = normalizar_nombre(row.get("AwayTeam"))
        gl, gv = n(row, "FTHG"), n(row, "FTAG")
        if not local or not visita or pd.isna(gl) or pd.isna(gv):
            continue

        hs, ass = n(row, "HS"), n(row, "AS")
        hst, ast = n(row, "HST"), n(row, "AST")
        hc, ac = n(row, "HC", 5.0), n(row, "AC", 5.0)
        hy, ay = n(row, "HY", 2.0), n(row, "AY", 2.0)
        hr, ar = n(row, "HR", 0.0), n(row, "AR", 0.0)
        hf, af = n(row, "HF", np.nan), n(row, "AF", np.nan)

        fecha = row.get("Date", "")
        fecha_dt = pd.to_datetime(fecha, dayfirst=True, errors="coerce")
        fecha_iso = "" if pd.isna(fecha_dt) else fecha_dt.strftime("%Y-%m-%d")

        # Apertura: promedios publicados al inicio; fallback a Bet365/Pinnacle prepartido.
        open_h = first_numeric(row, ["AvgH", "B365H", "PSH", "MaxH"])
        open_d = first_numeric(row, ["AvgD", "B365D", "PSD", "MaxD"])
        open_a = first_numeric(row, ["AvgA", "B365A", "PSA", "MaxA"])
        open_o25 = first_numeric(row, ["Avg>2.5", "B365>2.5", "P>2.5", "Max>2.5"])
        open_u25 = first_numeric(row, ["Avg<2.5", "B365<2.5", "P<2.5", "Max<2.5"])

        # Cierre: columnas C cuando existen. Si no existen, conserva la mejor prepartido disponible.
        close_h = first_numeric(row, ["AvgCH", "B365CH", "PSCH", "MaxCH"], open_h)
        close_d = first_numeric(row, ["AvgCD", "B365CD", "PSCD", "MaxCD"], open_d)
        close_a = first_numeric(row, ["AvgCA", "B365CA", "PSCA", "MaxCA"], open_a)
        close_o25 = first_numeric(row, ["AvgC>2.5", "B365C>2.5", "PC>2.5", "MaxC>2.5"], open_o25)
        close_u25 = first_numeric(row, ["AvgC<2.5", "B365C<2.5", "PC<2.5", "MaxC<2.5"], open_u25)

        p1o, pxo, p2o = no_vig_1x2(open_h, open_d, open_a)
        p1c, pxc, p2c = no_vig_1x2(close_h, close_d, close_a)

        rows.append({
            "Fecha": fecha_iso,
            "Local": local,
            "Visitante": visita,
            "Goles_Local": int(gl),
            "Goles_Visita": int(gv),
            "Tiros_Local": hs if not pd.isna(hs) else np.nan,
            "Tiros_Visita": ass if not pd.isna(ass) else np.nan,
            "Corners_Local": hc,
            "Corners_Visita": ac,
            "Tarjetas_Local": hy + 2.0 * hr,
            "Tarjetas_Visita": ay + 2.0 * ar,
            "Faltas_Local": hf,
            "Faltas_Visita": af,
            "xG_Local": xg_proxy(hs, hst, gl),
            "xG_Visita": xg_proxy(ass, ast, gv),
            "xG_Proxy_Local": xg_proxy(hs, hst, gl),
            "xG_Proxy_Visita": xg_proxy(ass, ast, gv),
            "TirosGol_Local": hst if not pd.isna(hst) else 4.0,
            "TirosGol_Visita": ast if not pd.isna(ast) else 4.0,
            "Atajadas_Local": max(0.0, (ast if not pd.isna(ast) else 3.0) - gv),
            "Atajadas_Visita": max(0.0, (hst if not pd.isna(hst) else 3.0) - gl),

            # Compatibilidad: Cuota_* representa mercado de apertura utilizable por producción temprana.
            "Cuota_1": open_h,
            "Cuota_X": open_d,
            "Cuota_2": open_a,
            "Cuota_Over25": open_o25,
            "Cuota_Under25": open_u25,

            "Apertura_1": open_h,
            "Apertura_X": open_d,
            "Apertura_2": open_a,
            "Cierre_1": close_h,
            "Cierre_X": close_d,
            "Cierre_2": close_a,
            "Apertura_Over25": open_o25,
            "Apertura_Under25": open_u25,
            "Cierre_Over25": close_o25,
            "Cierre_Under25": close_u25,
            "P_Apertura_1": p1o,
            "P_Apertura_X": pxo,
            "P_Apertura_2": p2o,
            "P_Cierre_1": p1c,
            "P_Cierre_X": pxc,
            "P_Cierre_2": p2c,
            "Movimiento_1_pp": (p1c - p1o) * 100.0 if np.isfinite(p1c) and np.isfinite(p1o) else np.nan,
            "Movimiento_X_pp": (pxc - pxo) * 100.0 if np.isfinite(pxc) and np.isfinite(pxo) else np.nan,
            "Movimiento_2_pp": (p2c - p2o) * 100.0 if np.isfinite(p2c) and np.isfinite(p2o) else np.nan,

            "Arbitro": str(row.get("Referee", "Desconocido") or "Desconocido"),
            "Fuente": "football-data.co.uk",
            "Fuente_xG": "proxy_shots",
            "StatsBomb_match_id": np.nan,
        })
    return pd.DataFrame(rows)


def aplicar_statsbomb(out):
    """Sustituye el proxy por xG real sólo en matches exactos del cache abierto."""
    if out is None or out.empty or not os.path.exists(STATSBOMB_CACHE):
        return out, 0
    try:
        sb = pd.read_csv(STATSBOMB_CACHE)
    except Exception:
        return out, 0
    required = {"Fecha", "Local_norm", "Visitante_norm", "xG_Real_Local", "xG_Real_Visita"}
    if sb.empty or not required.issubset(sb.columns):
        return out, 0

    x = out.copy()
    x["_Local_norm"] = x["Local"].map(key_nombre)
    x["_Visitante_norm"] = x["Visitante"].map(key_nombre)
    sb = sb.copy()
    sb["Fecha"] = sb["Fecha"].astype(str)
    keep = [c for c in [
        "Fecha", "Local_norm", "Visitante_norm", "StatsBomb_match_id",
        "xG_Real_Local", "xG_Real_Visita", "Tiros_SB_Local", "Tiros_SB_Visita",
        "TirosGol_SB_Local", "TirosGol_SB_Visita", "Fuente_xG_Real",
    ] if c in sb.columns]
    sb = sb[keep].drop_duplicates(subset=["Fecha", "Local_norm", "Visitante_norm"], keep="last")
    merged = x.merge(
        sb,
        how="left",
        left_on=["Fecha", "_Local_norm", "_Visitante_norm"],
        right_on=["Fecha", "Local_norm", "Visitante_norm"],
        suffixes=("", "_SB"),
    )
    mask = pd.to_numeric(merged.get("xG_Real_Local"), errors="coerce").notna() & pd.to_numeric(merged.get("xG_Real_Visita"), errors="coerce").notna()
    if mask.any():
        merged.loc[mask, "xG_Local"] = pd.to_numeric(merged.loc[mask, "xG_Real_Local"], errors="coerce")
        merged.loc[mask, "xG_Visita"] = pd.to_numeric(merged.loc[mask, "xG_Real_Visita"], errors="coerce")
        merged.loc[mask, "Fuente_xG"] = "StatsBomb Open Data"
        if "StatsBomb_match_id_SB" in merged.columns:
            merged.loc[mask, "StatsBomb_match_id"] = merged.loc[mask, "StatsBomb_match_id_SB"]
        elif "StatsBomb_match_id_y" in merged.columns:
            merged.loc[mask, "StatsBomb_match_id"] = merged.loc[mask, "StatsBomb_match_id_y"]
    drop_cols = [c for c in ["_Local_norm", "_Visitante_norm", "Local_norm", "Visitante_norm", "StatsBomb_match_id_SB", "StatsBomb_match_id_y"] if c in merged.columns]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    return merged, int(mask.sum())


def procesar_liga(nombre, codigo, salida):
    partes = []
    for season in SEASON_CODES:
        bruto = descargar_csv(season, codigo)
        limpio = transformar(bruto)
        print(f"{nombre} {season}: {len(limpio)} partidos")
        if not limpio.empty:
            partes.append(limpio)

    if not partes:
        raise RuntimeError(f"No se pudo descargar histórico para {nombre}")

    out = pd.concat(partes, ignore_index=True)
    out = out.drop_duplicates(subset=["Fecha", "Local", "Visitante"], keep="last")
    out = out.sort_values(["Fecha", "Local", "Visitante"]).reset_index(drop=True)
    out, sb_n = aplicar_statsbomb(out)
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    out.to_csv(salida, index=False)
    coverage = int(out[["Apertura_1", "Apertura_X", "Apertura_2", "Cierre_1", "Cierre_X", "Cierre_2"]].notna().all(axis=1).sum())
    print(f"OK {salida}: {len(out)} registros | apertura+cierre 1X2: {coverage} | StatsBomb xG real: {sb_n}")
    return len(out)


def main():
    total = 0
    for nombre, cfg in LIGAS.items():
        total += procesar_liga(nombre, cfg["codigo"], cfg["archivo"])
    if total < 3000:
        raise RuntimeError(f"Histórico insuficiente: {total} registros")
    print(f"Históricos completos: {total} registros")


if __name__ == "__main__":
    main()
