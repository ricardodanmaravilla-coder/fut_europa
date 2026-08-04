import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class PredictorMLEuropa:
    def __init__(self):
        # Inicializamos 4 modelos de Random Forest independientes para no mezclar sesgos
        self.modelo_1x2 = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42, class_weight='balanced')
        self.modelo_goles = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.modelo_corners = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.modelo_tarjetas = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.stats_equipos = {}
        self.entrenado = False

    def _calcular_perfiles_equipos(self, df):
        """Calcula los promedios históricos de métricas avanzadas (xG, tiros, atajadas) por equipo"""
        equipos = set(df['Local'].unique()).union(set(df['Visitante'].unique()))
        
        for eq in equipos:
            df_loc = df[df['Local'] == eq]
            df_vis = df[df['Visitante'] == eq]
            
            # Promedios Ponderados (Mitad de local, mitad de visita)
            xg = (df_loc['xG_Local'].mean() + df_vis['xG_Visita'].mean()) / 2
            tiros = (df_loc['TirosGol_Local'].mean() + df_vis['TirosGol_Visita'].mean()) / 2
            atajadas = (df_loc['Atajadas_Local'].mean() + df_vis['Atajadas_Visita'].mean()) / 2
            corners = (df_loc['Corners_Local'].mean() + df_vis['Corners_Visita'].mean()) / 2
            tarjetas = (df_loc['Tarjetas_Local'].mean() + df_vis['Tarjetas_Visita'].mean()) / 2
            
            self.stats_equipos[eq] = {
                'xg_prom': xg if pd.notna(xg) else 1.2,
                'tiros_prom': tiros if pd.notna(tiros) else 4.0,
                'atajadas_prom': atajadas if pd.notna(atajadas) else 3.0,
                'corners_prom': corners if pd.notna(corners) else 4.5,
                'tarjetas_prom': tarjetas if pd.notna(tarjetas) else 2.0
            }

    def entrenar(self, df_historico):
        if df_historico is None or df_historico.empty:
            return False

        # 1. Extraer los perfiles matemáticos de todos los equipos
        self._calcular_perfiles_equipos(df_historico)

        # 2. Preparar los datos de entrenamiento (Features y Targets)
        X, y_1x2, y_goles, y_corners, y_tarjetas = [], [], [], [], []

        for _, row in df_historico.iterrows():
            loc = row['Local']
            vis = row['Visitante']
            
            if pd.isna(loc) or pd.isna(vis) or loc not in self.stats_equipos or vis not in self.stats_equipos:
                continue
                
            # Extraer estadísticas del partido real para los Targets (Lo que realmente pasó)
            g_l, g_v = row['Goles_Local'], row['Goles_Visita']
            tot_goles = g_l + g_v
            tot_corners = row['Corners_Local'] + row['Corners_Visita']
            tot_tarjetas = row['Tarjetas_Local'] + row['Tarjetas_Visita']

            # Target 1X2: 0 (Gana Visita), 1 (Empate), 2 (Gana Local)
            if g_l > g_v: target_1x2 = 2
            elif g_l == g_v: target_1x2 = 1
            else: target_1x2 = 0

            # Targets Over/Under
            target_goles = 1 if tot_goles > 2.5 else 0
            target_corners = 1 if tot_corners > 9.5 else 0
            target_tarjetas = 1 if tot_tarjetas > 4.5 else 0

            # Features (Variables predictoras basadas en los promedios del equipo para evitar data leakage)
            stats_l = self.stats_equipos[loc]
            stats_v = self.stats_equipos[vis]
            
            # Matriz de características (Cómo se ve el choque en papel)
            features = [
                stats_l['xg_prom'], stats_v['xg_prom'],                 # Fuerza Ofensiva (Expected Goals)
                stats_l['tiros_prom'], stats_v['tiros_prom'],           # Volumen de Llegada
                stats_l['atajadas_prom'], stats_v['atajadas_prom'],     # Solidez Defensiva / Portero
                stats_l['corners_prom'], stats_v['corners_prom'],       # Presión en bandas
                stats_l['tarjetas_prom'], stats_v['tarjetas_prom']      # Indisciplina
            ]
            
            X.append(features)
            y_1x2.append(target_1x2)
            y_goles.append(target_goles)
            y_corners.append(target_corners)
            y_tarjetas.append(target_tarjetas)

        if len(X) < 20: # Requiere un mínimo de partidos para entrenar
            return False

        # 3. Entrenar los 4 modelos simultáneamente
        X = np.array(X)
        self.modelo_1x2.fit(X, y_1x2)
        self.modelo_goles.fit(X, y_goles)
        self.modelo_corners.fit(X, y_corners)
        self.modelo_tarjetas.fit(X, y_tarjetas)
        
        self.entrenado = True
        return True

    def predecir_mercados_completos(self, local, visita, goles_sim_l, goles_sim_v, elo_local, elo_visita):
        """Genera las probabilidades usando el modelo entrenado, inyectando variables del partido actual"""
        if not self.entrenado:
            return {"error": "El modelo no está entrenado (Faltan datos históricos)."}
            
        stats_l = self.stats_equipos.get(local, {'xg_prom':1.2, 'tiros_prom':4.0, 'atajadas_prom':3.0, 'corners_prom':4.5, 'tarjetas_prom':2.0})
        stats_v = self.stats_equipos.get(visita, {'xg_prom':1.2, 'tiros_prom':4.0, 'atajadas_prom':3.0, 'corners_prom':4.5, 'tarjetas_prom':2.0})
        
        # Construimos el array exacto que aprendió el modelo
        X_pred = np.array([[
            stats_l['xg_prom'], stats_v['xg_prom'],
            stats_l['tiros_prom'], stats_v['tiros_prom'],
            stats_l['atajadas_prom'], stats_v['atajadas_prom'],
            stats_l['corners_prom'], stats_v['corners_prom'],
            stats_l['tarjetas_prom'], stats_v['tarjetas_prom']
        ]])

        # Extraemos probabilidades de los árboles de decisión
        prob_1x2 = self.modelo_1x2.predict_proba(X_pred)[0]
        prob_goles = self.modelo_goles.predict_proba(X_pred)[0]
        prob_corners = self.modelo_corners.predict_proba(X_pred)[0]
        prob_tarjetas = self.modelo_tarjetas.predict_proba(X_pred)[0]

        # Asegurar el formato (El índice 1 siempre es la clase Positiva/Over)
        return {
            "Resultado_1X2": {
                "Gana Local": round(prob_1x2[2] * 100, 1) if len(prob_1x2) == 3 else 0.0,
                "Empate": round(prob_1x2[1] * 100, 1) if len(prob_1x2) == 3 else 0.0,
                "Gana Visita": round(prob_1x2[0] * 100, 1) if len(prob_1x2) == 3 else 0.0
            },
            "Goles_Over_Under": {
                "Over 2.5": round(prob_goles[1] * 100, 1) if len(prob_goles) > 1 else 0.0,
                "Under 2.5": round(prob_goles[0] * 100, 1)
            },
            "Corners_Totales": {
                "Over 9.5 Corners": round(prob_corners[1] * 100, 1) if len(prob_corners) > 1 else 0.0,
                "Under 9.5 Corners": round(prob_corners[0] * 100, 1)
            },
            "Tarjetas_Totales": {
                "Over 4.5 Tarjetas": round(prob_tarjetas[1] * 100, 1) if len(prob_tarjetas) > 1 else 0.0,
                "Under 4.5 Tarjetas": round(prob_tarjetas[0] * 100, 1)
            }
        }
