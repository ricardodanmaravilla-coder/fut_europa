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


def normalizar_nombre(nombre):
    txt = str(nombre or "").strip()
    return unicodedata.normalize("NFKD", txt).encode("ASCII", "ignore").decode("utf-8").strip()


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
            "TirosGol_Local": hst if not pd.isna(hst) else 4.0,
            "TirosGol_Visita": ast if not pd.isna(ast) else 4.0,
            "Atajadas_Local": max(0.0, (ast if not pd.isna(ast) else 3.0) - gv),
            "Atajadas_Visita": max(0.0, (hst if not pd.isna(hst) else 3.0) - gl),

            # Compatibilidad: Cuota_* sigue representando mercado prepartido utilizable por producción.
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
        })
    return pd.DataFrame(rows)


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
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    out.to_csv(salida, index=False)
    coverage = int(out[["Apertura_1", "Apertura_X", "Apertura_2", "Cierre_1", "Cierre_X", "Cierre_2"]].notna().all(axis=1).sum())
    print(f"OK {salida}: {len(out)} registros | apertura+cierre 1X2: {coverage}")
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
