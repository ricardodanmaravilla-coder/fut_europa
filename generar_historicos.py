import io
import os
import unicodedata

import numpy as np
import pandas as pd
import requests

# Fuente primaria gratuita para histórico cuantitativo.
# Códigos: E0 Premier, SP1 La Liga, I1 Serie A, D1 Bundesliga, F1 Ligue 1.
LIGAS = {
    "Premier League": {"codigo": "E0", "archivo": "data/historico_premier.csv"},
    "La Liga": {"codigo": "SP1", "archivo": "data/historico_laliga.csv"},
    "Serie A": {"codigo": "I1", "archivo": "data/historico_seriea.csv"},
    "Bundesliga": {"codigo": "D1", "archivo": "data/historico_bundesliga.csv"},
    "Ligue 1": {"codigo": "F1", "archivo": "data/historico_ligue1.csv"},
}

# Temporadas empezando en 2020-21 y terminando en 2026-27.
SEASON_CODES = ["2021", "2122", "2223", "2324", "2425", "2526", "2627"]
BASE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


def normalizar_nombre(nombre):
    txt = str(nombre or "").strip()
    return unicodedata.normalize("NFKD", txt).encode("ASCII", "ignore").decode("utf-8").strip()


def n(row, key, default=np.nan):
    val = pd.to_numeric(row.get(key), errors="coerce")
    return default if pd.isna(val) else float(val)


def xg_proxy(shots, shots_on_target, goals):
    """Proxy conservador cuando la fuente no publica xG real.

    Se construye sólo con variables preexistentes en el partido para que la app pueda
    funcionar sin inventar una supuesta métrica oficial. El CSV conserva Fuente_xG.
    """
    s = 0.0 if pd.isna(shots) else float(shots)
    sot = 0.0 if pd.isna(shots_on_target) else float(shots_on_target)
    g = 0.0 if pd.isna(goals) else float(goals)
    return round(float(np.clip(0.055 * s + 0.18 * sot + 0.10 * g, 0.20, 4.00)), 2)


def descargar_csv(season_code, league_code):
    url = BASE.format(season=season_code, league=league_code)
    r = requests.get(url, timeout=20, headers={"User-Agent": "fut-europa/1.0"})
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

        fecha = row.get("Date", "")
        fecha_dt = pd.to_datetime(fecha, dayfirst=True, errors="coerce")
        fecha_iso = "" if pd.isna(fecha_dt) else fecha_dt.strftime("%Y-%m-%d")

        rows.append({
            "Fecha": fecha_iso,
            "Local": local,
            "Visitante": visita,
            "Goles_Local": int(gl),
            "Goles_Visita": int(gv),
            "Corners_Local": hc,
            "Corners_Visita": ac,
            "Tarjetas_Local": hy + 2.0 * hr,
            "Tarjetas_Visita": ay + 2.0 * ar,
            "xG_Local": xg_proxy(hs, hst, gl),
            "xG_Visita": xg_proxy(ass, ast, gv),
            "TirosGol_Local": hst if not pd.isna(hst) else 4.0,
            "TirosGol_Visita": ast if not pd.isna(ast) else 4.0,
            # La fuente no incluye atajadas en todas las temporadas; se aproxima con tiros a puerta recibidos - goles.
            "Atajadas_Local": max(0.0, (ast if not pd.isna(ast) else 3.0) - gv),
            "Atajadas_Visita": max(0.0, (hst if not pd.isna(hst) else 3.0) - gl),
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
    print(f"OK {salida}: {len(out)} registros")
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
