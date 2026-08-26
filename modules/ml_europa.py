import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


DEFAULT = {
    "xg_for": 1.25,
    "xg_against": 1.25,
    "shots_for": 4.0,
    "saves": 3.0,
    "corners_for": 4.8,
    "cards": 2.0,
    "games": 0,
}


class PredictorMLEuropa:
    """Modelos prepartido sin leakage temporal.

    Cada fila de entrenamiento se construye con estadísticas disponibles ANTES
    del partido. El resultado de ese partido sólo se incorpora al estado después
    de crear sus features/targets.
    """

    def __init__(self):
        self.modelo_1x2 = RandomForestClassifier(
            n_estimators=250, max_depth=7, min_samples_leaf=8,
            random_state=42, class_weight="balanced_subsample"
        )
        self.modelo_goles = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=8, random_state=43
        )
        self.modelo_corners = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=8, random_state=44
        )
        self.modelo_tarjetas = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=8, random_state=45
        )
        self.stats_equipos = {}
        self.entrenado = False
        self.n_train = 0

    @staticmethod
    def _safe(v, default):
        try:
            return default if pd.isna(v) else float(v)
        except Exception:
            return default

    @staticmethod
    def _new_state():
        return {
            "games": 0,
            "xg_for_sum": 0.0,
            "xg_against_sum": 0.0,
            "shots_for_sum": 0.0,
            "saves_sum": 0.0,
            "corners_for_sum": 0.0,
            "cards_sum": 0.0,
        }

    @staticmethod
    def _profile(state):
        n = state.get("games", 0)
        if n <= 0:
            return DEFAULT.copy()
        return {
            "xg_for": state["xg_for_sum"] / n,
            "xg_against": state["xg_against_sum"] / n,
            "shots_for": state["shots_for_sum"] / n,
            "saves": state["saves_sum"] / n,
            "corners_for": state["corners_for_sum"] / n,
            "cards": state["cards_sum"] / n,
            "games": n,
        }

    @staticmethod
    def _elo_expected(a, b):
        return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))

    def _features(self, pl, pv, elo_l, elo_v):
        exp_g_l = max(0.20, (pl["xg_for"] + pv["xg_against"]) / 2.0)
        exp_g_v = max(0.20, (pv["xg_for"] + pl["xg_against"]) / 2.0)
        return [
            pl["xg_for"], pv["xg_for"],
            pl["xg_against"], pv["xg_against"],
            pl["shots_for"], pv["shots_for"],
            pl["saves"], pv["saves"],
            pl["corners_for"], pv["corners_for"],
            pl["cards"], pv["cards"],
            exp_g_l, exp_g_v,
            (float(elo_l) - float(elo_v)) / 400.0,
        ]

    def entrenar(self, df_historico):
        self.entrenado = False
        self.stats_equipos = {}
        if df_historico is None or df_historico.empty:
            return False

        df = df_historico.copy()
        if "Fecha" in df.columns:
            df["_fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
            df = df.sort_values("_fecha", kind="stable")

        states = {}
        ratings = {}
        X, y_1x2, y_goles, y_corners, y_tarjetas = [], [], [], [], []

        for _, row in df.iterrows():
            loc, vis = row.get("Local"), row.get("Visitante")
            if pd.isna(loc) or pd.isna(vis):
                continue
            loc, vis = str(loc).strip(), str(vis).strip()
            sl = states.setdefault(loc, self._new_state())
            sv = states.setdefault(vis, self._new_state())
            pl, pv = self._profile(sl), self._profile(sv)
            elo_l, elo_v = ratings.get(loc, 1500.0), ratings.get(vis, 1500.0)

            g_l = self._safe(row.get("Goles_Local"), np.nan)
            g_v = self._safe(row.get("Goles_Visita"), np.nan)
            if pd.isna(g_l) or pd.isna(g_v):
                continue

            # Exigimos una pequeña historia previa para que la observación sea predecible.
            if pl["games"] >= 3 and pv["games"] >= 3:
                X.append(self._features(pl, pv, elo_l, elo_v))
                y_1x2.append(2 if g_l > g_v else (1 if g_l == g_v else 0))
                y_goles.append(int(g_l + g_v > 2.5))
                y_corners.append(int(
                    self._safe(row.get("Corners_Local"), 0.0)
                    + self._safe(row.get("Corners_Visita"), 0.0) > 9.5
                ))
                y_tarjetas.append(int(
                    self._safe(row.get("Tarjetas_Local"), 0.0)
                    + self._safe(row.get("Tarjetas_Visita"), 0.0) > 4.5
                ))

            # Actualizar estados SOLAMENTE después de crear la fila prepartido.
            xgl = self._safe(row.get("xG_Local"), g_l)
            xgv = self._safe(row.get("xG_Visita"), g_v)
            for state, xgf, xga, shots, saves, corners, cards in [
                (sl, xgl, xgv, row.get("TirosGol_Local"), row.get("Atajadas_Local"), row.get("Corners_Local"), row.get("Tarjetas_Local")),
                (sv, xgv, xgl, row.get("TirosGol_Visita"), row.get("Atajadas_Visita"), row.get("Corners_Visita"), row.get("Tarjetas_Visita")),
            ]:
                state["games"] += 1
                state["xg_for_sum"] += self._safe(xgf, 1.25)
                state["xg_against_sum"] += self._safe(xga, 1.25)
                state["shots_for_sum"] += self._safe(shots, 4.0)
                state["saves_sum"] += self._safe(saves, 3.0)
                state["corners_for_sum"] += self._safe(corners, 4.8)
                state["cards_sum"] += self._safe(cards, 2.0)

            exp_l = self._elo_expected(elo_l, elo_v)
            score_l = 1.0 if g_l > g_v else (0.5 if g_l == g_v else 0.0)
            delta = 24.0 * (score_l - exp_l)
            ratings[loc] = elo_l + delta
            ratings[vis] = elo_v - delta

        if len(X) < 100 or len(set(y_1x2)) < 3:
            return False

        X = np.asarray(X, dtype=float)
        self.modelo_1x2.fit(X, y_1x2)
        self.modelo_goles.fit(X, y_goles)
        self.modelo_corners.fit(X, y_corners)
        self.modelo_tarjetas.fit(X, y_tarjetas)
        self.stats_equipos = {team: self._profile(st) for team, st in states.items()}
        self.n_train = len(X)
        self.entrenado = True
        return True

    @staticmethod
    def _class_prob(model, probs, cls):
        classes = list(model.classes_)
        return float(probs[classes.index(cls)]) if cls in classes else 0.0

    def predecir_mercados_completos(self, local, visita, goles_sim_l=None, goles_sim_v=None, elo_local=1500, elo_visita=1500):
        if not self.entrenado:
            return {"error": "El modelo no está entrenado (faltan datos históricos)."}

        pl = self.stats_equipos.get(local, DEFAULT)
        pv = self.stats_equipos.get(visita, DEFAULT)
        x = np.asarray([self._features(pl, pv, elo_local, elo_visita)], dtype=float)

        p1 = self.modelo_1x2.predict_proba(x)[0]
        pg = self.modelo_goles.predict_proba(x)[0]
        pc = self.modelo_corners.predict_proba(x)[0]
        pt = self.modelo_tarjetas.predict_proba(x)[0]

        p_home = self._class_prob(self.modelo_1x2, p1, 2) * 100
        p_draw = self._class_prob(self.modelo_1x2, p1, 1) * 100
        p_away = self._class_prob(self.modelo_1x2, p1, 0) * 100
        p_go = self._class_prob(self.modelo_goles, pg, 1) * 100
        p_co = self._class_prob(self.modelo_corners, pc, 1) * 100
        p_to = self._class_prob(self.modelo_tarjetas, pt, 1) * 100

        return {
            "Resultado_1X2": {"Gana Local": round(p_home, 1), "Empate": round(p_draw, 1), "Gana Visita": round(p_away, 1)},
            "Goles_Over_Under": {"Over 2.5": round(p_go, 1), "Under 2.5": round(100-p_go, 1)},
            "Corners_Totales": {"Over 9.5 Corners": round(p_co, 1), "Under 9.5 Corners": round(100-p_co, 1)},
            "Tarjetas_Totales": {"Over 4.5 Tarjetas": round(p_to, 1), "Under 4.5 Tarjetas": round(100-p_to, 1)},
            "Meta": {"train_rows": self.n_train, "pregame_only": True},
        }
