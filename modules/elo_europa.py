import pandas as pd
import numpy as np

class SistemaEloEuropa:
    def __init__(self, k=32, base=1500):
        self.k = k
        self.base = base
        self.ratings = {}

    def probabilidad_esperada(self, elo_a, elo_b):
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    def actualizar_ratings(self, df_historico):
        if df_historico is None or df_historico.empty:
            return pd.DataFrame(columns=["Equipo", "ELO_Rating"])

        self.ratings = {}
        
        # Ordenar cronológicamente si hay columna fecha
        if 'Fecha' in df_historico.columns:
            df = df_historico.sort_values(by='Fecha').copy()
        else:
            df = df_historico.copy()

        for _, row in df.iterrows():
            loc = row['Local']
            vis = row['Visitante']
            g_loc = row['Goles_Local']
            g_vis = row['Goles_Visita']

            if pd.isna(loc) or pd.isna(vis) or pd.isna(g_loc) or pd.isna(g_vis):
                continue

            if loc not in self.ratings: self.ratings[loc] = self.base
            if vis not in self.ratings: self.ratings[vis] = self.base

            elo_l = self.ratings[loc]
            elo_v = self.ratings[vis]

            exp_l = self.probabilidad_esperada(elo_l, elo_v)
            exp_v = 1.0 - exp_l

            if g_loc > g_vis:
                res_l, res_v = 1.0, 0.0
            elif g_loc < g_vis:
                res_l, res_v = 0.0, 1.0
            else:
                res_l, res_v = 0.5, 0.5

            self.ratings[loc] = elo_l + self.k * (res_l - exp_l)
            self.ratings[vis] = elo_v + self.k * (res_v - exp_v)

        # Convertir a DataFrame ordenado
        df_ranking = pd.DataFrame(list(self.ratings.items()), columns=["Equipo", "ELO_Rating"])
        df_ranking = df_ranking.sort_values(by="ELO_Rating", ascending=False).reset_index(drop=True)
        return df_ranking
