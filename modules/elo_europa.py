import pandas as pd
import numpy as np

HOME_ADVANTAGE = 55.0
BASE_K = 22.0


class SistemaEloEuropa:
    """Elo coherente con el Elo usado por PredictorMLEuropa.

    La ventaja local sólo interviene en la probabilidad esperada; los ratings
    almacenados siguen siendo neutrales. K aumenta moderadamente con el margen,
    exactamente como durante el entrenamiento ML.
    """
    def __init__(self, k=BASE_K, base=1500, home_advantage=HOME_ADVANTAGE):
        self.k = float(k)
        self.base = float(base)
        self.home_advantage = float(home_advantage)
        self.ratings = {}

    def probabilidad_esperada(self, elo_local, elo_visita):
        return 1.0 / (1.0 + 10.0 ** ((float(elo_visita) - (float(elo_local) + self.home_advantage)) / 400.0))

    def actualizar_ratings(self, df_historico):
        if df_historico is None or df_historico.empty:
            return pd.DataFrame(columns=["Equipo", "ELO_Rating"])

        self.ratings = {}
        df = df_historico.copy()
        if "Fecha" in df.columns:
            df["_fecha_elo"] = pd.to_datetime(df["Fecha"], errors="coerce", format="%Y-%m-%d")
            df = df.sort_values("_fecha_elo", kind="stable")

        for _, row in df.iterrows():
            loc, vis = row.get("Local"), row.get("Visitante")
            g_loc, g_vis = row.get("Goles_Local"), row.get("Goles_Visita")
            if pd.isna(loc) or pd.isna(vis) or pd.isna(g_loc) or pd.isna(g_vis):
                continue
            loc, vis = str(loc).strip(), str(vis).strip()
            try:
                g_loc, g_vis = float(g_loc), float(g_vis)
            except Exception:
                continue

            elo_l = self.ratings.get(loc, self.base)
            elo_v = self.ratings.get(vis, self.base)
            exp_l = self.probabilidad_esperada(elo_l, elo_v)
            score_l = 1.0 if g_loc > g_vis else (0.5 if g_loc == g_vis else 0.0)
            k_match = self.k * (1.0 + 0.12 * min(abs(g_loc - g_vis), 4.0))
            delta = k_match * (score_l - exp_l)
            self.ratings[loc] = elo_l + delta
            self.ratings[vis] = elo_v - delta

        out = pd.DataFrame(list(self.ratings.items()), columns=["Equipo", "ELO_Rating"])
        if not out.empty:
            out = out.sort_values("ELO_Rating", ascending=False).reset_index(drop=True)
        return out
