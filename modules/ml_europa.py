import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import log_loss

DEFAULT = {
    "xg_for": 1.25, "xg_against": 1.25, "goals_for": 1.30, "goals_against": 1.30,
    "sot_for": 4.0, "sot_against": 4.0, "corners_for": 4.8, "corners_against": 4.8,
    "cards": 2.0, "points": 1.35, "games": 0, "rest_days": 7.0,
    "xt_for": 0.0, "xt_against": 0.0, "ppda_for": 12.0, "ppda_against": 12.0,
    "event_games": 0, "event_coverage": 0.0,
}
EMA_ALPHA = 0.28


def _normalize_rows(p):
    p = np.asarray(p, dtype=float); p = np.clip(p, 1e-6, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _temperature(p, t):
    p = _normalize_rows(p); z = np.power(p, 1.0 / max(float(t), 0.15))
    return z / z.sum(axis=1, keepdims=True)


def _no_vig_1x2(h, d, a):
    try:
        odds=np.array([float(a),float(d),float(h)],dtype=float)
        if np.any(~np.isfinite(odds)) or np.any(odds<=1.0): return None
        q=1.0/odds; return q/q.sum()
    except Exception: return None


class PredictorMLEuropa:
    def __init__(self, use_event_features=False):
        self.use_event_features=bool(use_event_features)
        self.modelo_1x2=ExtraTreesClassifier(n_estimators=350,max_depth=10,min_samples_leaf=7,max_features=0.80,random_state=42,class_weight="balanced")
        self.modelo_goles=RandomForestClassifier(n_estimators=260,max_depth=8,min_samples_leaf=7,max_features=0.8,random_state=43,class_weight="balanced_subsample")
        self.modelo_corners=RandomForestClassifier(n_estimators=240,max_depth=8,min_samples_leaf=8,max_features=0.8,random_state=44,class_weight="balanced_subsample")
        self.modelo_tarjetas=RandomForestClassifier(n_estimators=240,max_depth=8,min_samples_leaf=8,max_features=0.8,random_state=45,class_weight="balanced_subsample")
        self.stats_equipos={}; self.raw_states={}; self.entrenado=False; self.n_train=0
        self.temp_1x2=self.temp_goles=self.temp_corners=self.temp_tarjetas=1.0
        self.market_model_weight=1.0; self.calibration_rows=0

    @staticmethod
    def _safe(v, default):
        try: return default if pd.isna(v) else float(v)
        except Exception: return default

    @staticmethod
    def _finite(v):
        try: return pd.notna(v) and np.isfinite(float(v))
        except Exception: return False

    @staticmethod
    def _new_state():
        d={"games":0,"event_games":0,"last_date":None}
        for k in ["xg_for","xg_against","goals_for","goals_against","sot_for","sot_against","corners_for","corners_against","cards","points","xt_for","xt_against","ppda_for","ppda_against"]:
            d[k]=None; d[f"sum_{k}"]=0.0
        return d

    @staticmethod
    def _ema(old,new): return float(new) if old is None else EMA_ALPHA*float(new)+(1.0-EMA_ALPHA)*float(old)

    def _profile(self,state,match_date=None):
        if not state or int(state.get("games",0))<=0: return DEFAULT.copy()
        n=int(state.get("games",0)); en=int(state.get("event_games",0))
        def blend(key,default,den=None):
            d=max(int(den if den is not None else n),1); long=state.get(f"sum_{key}",0.0)/d; recent=state.get(key)
            if recent is None or not np.isfinite(recent): recent=long
            return 0.60*float(recent)+0.40*float(long if np.isfinite(long) else default)
        rest=7.0
        if match_date is not None and state.get("last_date") is not None:
            try: rest=float(np.clip((match_date-state["last_date"]).days,2,21))
            except Exception: pass
        return {"xg_for":blend("xg_for",1.25),"xg_against":blend("xg_against",1.25),"goals_for":blend("goals_for",1.30),"goals_against":blend("goals_against",1.30),
                "sot_for":blend("sot_for",4.0),"sot_against":blend("sot_against",4.0),"corners_for":blend("corners_for",4.8),"corners_against":blend("corners_against",4.8),
                "cards":blend("cards",2.0),"points":blend("points",1.35),"games":n,"rest_days":rest,
                "xt_for":blend("xt_for",0.0,en) if en else 0.0,"xt_against":blend("xt_against",0.0,en) if en else 0.0,
                "ppda_for":blend("ppda_for",12.0,en) if en else 12.0,"ppda_against":blend("ppda_against",12.0,en) if en else 12.0,
                "event_games":en,"event_coverage":min(en/5.0,1.0)}

    @staticmethod
    def _elo_expected(a,b,home_adv=55.0): return 1.0/(1.0+10.0**((b-(a+home_adv))/400.0))

    def _features(self,pl,pv,elo_l,elo_v):
        attack_l=0.55*pl["xg_for"]+0.25*pl["goals_for"]+0.20*(pl["sot_for"]/3.2); attack_v=0.55*pv["xg_for"]+0.25*pv["goals_for"]+0.20*(pv["sot_for"]/3.2)
        defense_l=0.60*pl["xg_against"]+0.25*pl["goals_against"]+0.15*(pl["sot_against"]/3.2); defense_v=0.60*pv["xg_against"]+0.25*pv["goals_against"]+0.15*(pv["sot_against"]/3.2)
        exp_l=max(0.20,0.58*attack_l+0.42*defense_v); exp_v=max(0.20,0.58*attack_v+0.42*defense_l)
        x=[pl["xg_for"],pv["xg_for"],pl["xg_against"],pv["xg_against"],pl["goals_for"],pv["goals_for"],pl["goals_against"],pv["goals_against"],pl["sot_for"],pv["sot_for"],pl["sot_against"],pv["sot_against"],pl["corners_for"],pv["corners_for"],pl["corners_against"],pv["corners_against"],pl["cards"],pv["cards"],pl["points"],pv["points"],pl["rest_days"],pv["rest_days"],pl["rest_days"]-pv["rest_days"],exp_l,exp_v,exp_l-exp_v,(float(elo_l)+55.0-float(elo_v))/400.0,self._elo_expected(float(elo_l),float(elo_v)),np.log1p(pl["games"]),np.log1p(pv["games"])]
        if self.use_event_features:
            x += [pl["xt_for"],pv["xt_for"],pl["xt_against"],pv["xt_against"],pl["xt_for"]-pv["xt_for"],pl["ppda_for"],pv["ppda_for"],pl["ppda_against"],pv["ppda_against"],pl["ppda_for"]-pv["ppda_for"],pl["event_coverage"],pv["event_coverage"],np.log1p(pl["event_games"]),np.log1p(pv["event_games"])]
        return x

    @staticmethod
    def _best_temperature(model,X_cal,y_cal):
        if len(X_cal)<80: return 1.0
        raw=model.predict_proba(X_cal); classes=list(model.classes_); best_t,best_loss=1.0,float("inf")
        for t in [0.70,0.85,1.0,1.15,1.30,1.50,1.75,2.0]:
            try: loss=log_loss(y_cal,_temperature(raw,t),labels=classes)
            except Exception: continue
            if loss<best_loss: best_t,best_loss=t,loss
        return float(best_t)

    def _fit_with_temporal_calibration(self,model,X,y):
        n=len(X); cut=max(100,int(n*0.80)); cut=min(cut,n-80) if n>=220 else n
        if cut<n:
            model.fit(X[:cut],y[:cut]); t=self._best_temperature(model,X[cut:],y[cut:]); self.calibration_rows=max(self.calibration_rows,n-cut)
        else: t=1.0
        model.fit(X,y); return t

    def entrenar(self,df_historico):
        self.entrenado=False; self.stats_equipos={}; self.raw_states={}
        if df_historico is None or df_historico.empty: return False
        df=df_historico.copy(); df["_fecha"]=pd.to_datetime(df.get("Fecha"),errors="coerce",format="%Y-%m-%d"); df=df.sort_values("_fecha",kind="stable")
        states,ratings={},{}; X=[]; y1=[]; yg=[]; yc=[]; yt=[]; markets=[]
        for _,row in df.iterrows():
            loc,vis=row.get("Local"),row.get("Visitante")
            if pd.isna(loc) or pd.isna(vis): continue
            loc,vis=str(loc).strip(),str(vis).strip(); date=row.get("_fecha"); sl=states.setdefault(loc,self._new_state()); sv=states.setdefault(vis,self._new_state())
            pl,pv=self._profile(sl,date),self._profile(sv,date); elo_l,elo_v=ratings.get(loc,1500.0),ratings.get(vis,1500.0)
            gl=self._safe(row.get("Goles_Local"),np.nan); gv=self._safe(row.get("Goles_Visita"),np.nan)
            if pd.isna(gl) or pd.isna(gv): continue
            if pl["games"]>=4 and pv["games"]>=4:
                X.append(self._features(pl,pv,elo_l,elo_v)); y1.append(2 if gl>gv else (1 if gl==gv else 0)); yg.append(int(gl+gv>2.5)); yc.append(int(self._safe(row.get("Corners_Local"),0)+self._safe(row.get("Corners_Visita"),0)>9.5)); yt.append(int(self._safe(row.get("Tarjetas_Local"),0)+self._safe(row.get("Tarjetas_Visita"),0)>4.5)); markets.append(_no_vig_1x2(row.get("Cuota_1"),row.get("Cuota_X"),row.get("Cuota_2")))
            xgl,xgv=self._safe(row.get("xG_Local"),gl),self._safe(row.get("xG_Visita"),gv); sotl,sotv=self._safe(row.get("TirosGol_Local"),4.0),self._safe(row.get("TirosGol_Visita"),4.0); cl,cv=self._safe(row.get("Corners_Local"),4.8),self._safe(row.get("Corners_Visita"),4.8); cal,cav=self._safe(row.get("Tarjetas_Local"),2.0),self._safe(row.get("Tarjetas_Visita"),2.0); pts_l=3.0 if gl>gv else (1.0 if gl==gv else 0.0); pts_v=3.0 if gv>gl else (1.0 if gl==gv else 0.0)
            for st,xgf,xga,gf,ga,sf,sa,cf,ca,cards,pts in [(sl,xgl,xgv,gl,gv,sotl,sotv,cl,cv,cal,pts_l),(sv,xgv,xgl,gv,gl,sotv,sotl,cv,cl,cav,pts_v)]:
                vals={"xg_for":xgf,"xg_against":xga,"goals_for":gf,"goals_against":ga,"sot_for":sf,"sot_against":sa,"corners_for":cf,"corners_against":ca,"cards":cards,"points":pts}; st["games"]+=1
                for key,val in vals.items(): val=float(val); st[key]=self._ema(st.get(key),val); st[f"sum_{key}"]+=val
                if pd.notna(date): st["last_date"]=date
            xtl,xtv=row.get("xThreat_Wyscout_Local"),row.get("xThreat_Wyscout_Visita"); ppl,ppv=row.get("PPDA_Local"),row.get("PPDA_Visita")
            if self.use_event_features and self._finite(xtl) and self._finite(xtv) and self._finite(ppl) and self._finite(ppv):
                for st,xtf,xta,ppf,ppa in [(sl,xtl,xtv,ppl,ppv),(sv,xtv,xtl,ppv,ppl)]:
                    st["event_games"]+=1
                    for key,val in {"xt_for":xtf,"xt_against":xta,"ppda_for":ppf,"ppda_against":ppa}.items(): val=float(val); st[key]=self._ema(st.get(key),val); st[f"sum_{key}"]+=val
            exp_l=self._elo_expected(elo_l,elo_v); score_l=1.0 if gl>gv else (0.5 if gl==gv else 0.0); k=22.0*(1.0+0.12*min(abs(gl-gv),4.0)); delta=k*(score_l-exp_l); ratings[loc],ratings[vis]=elo_l+delta,elo_v-delta
        if len(X)<250 or len(set(y1))<3: return False
        X=np.asarray(X,dtype=float); y1,yg,yc,yt=map(np.asarray,(y1,yg,yc,yt)); self.temp_1x2=self._fit_with_temporal_calibration(self.modelo_1x2,X,y1); self.temp_goles=self._fit_with_temporal_calibration(self.modelo_goles,X,yg); self.temp_corners=self._fit_with_temporal_calibration(self.modelo_corners,X,yc); self.temp_tarjetas=self._fit_with_temporal_calibration(self.modelo_tarjetas,X,yt)
        cut=int(len(X)*0.80)
        if len(X)-cut>=80:
            tmp=ExtraTreesClassifier(n_estimators=280,max_depth=10,min_samples_leaf=7,max_features=0.80,random_state=142,class_weight="balanced"); tmp.fit(X[:cut],y1[:cut]); pm=_temperature(tmp.predict_proba(X[cut:]),self.temp_1x2); idx=[]; marr=[]; target=[]
            for j,mk in enumerate(markets[cut:]):
                if mk is not None: idx.append(j); marr.append(mk); target.append(y1[cut+j])
            if len(idx)>=60:
                pmv=pm[np.asarray(idx)]; marr=np.asarray(marr); target=np.asarray(target); best_w,best_loss=1.0,float("inf")
                for w in [0.25,0.40,0.55,0.70,0.85,1.0]:
                    loss=log_loss(target,_normalize_rows(w*pmv+(1-w)*marr),labels=[0,1,2])
                    if loss<best_loss: best_w,best_loss=w,loss
                self.market_model_weight=float(best_w)
        self.raw_states=states; self.stats_equipos={team:self._profile(st,None) for team,st in states.items()}; self.n_train=len(X); self.entrenado=True; return True

    @staticmethod
    def _ordered_probs(model,raw,wanted):
        classes=list(model.classes_); return np.asarray([float(raw[classes.index(c)]) if c in classes else 0.0 for c in wanted])

    def predecir_mercados_completos(self,local,visita,goles_sim_l=None,goles_sim_v=None,elo_local=1500,elo_visita=1500,cuotas_1x2=None,fecha_partido=None):
        if not self.entrenado: return {"error":"El modelo no está entrenado (faltan datos históricos)."}
        date=pd.to_datetime(fecha_partido,errors="coerce") if fecha_partido else None
        if date is not None and pd.isna(date): date=None
        pl=self._profile(self.raw_states.get(local),date) if local in self.raw_states else DEFAULT; pv=self._profile(self.raw_states.get(visita),date) if visita in self.raw_states else DEFAULT; x=np.asarray([self._features(pl,pv,elo_local,elo_visita)],dtype=float)
        raw1=_temperature(self.modelo_1x2.predict_proba(x),self.temp_1x2)[0]; p1=self._ordered_probs(self.modelo_1x2,raw1,[0,1,2]); used_market=False
        if cuotas_1x2:
            mk=_no_vig_1x2(cuotas_1x2.get("1"),cuotas_1x2.get("X"),cuotas_1x2.get("2"))
            if mk is not None and self.market_model_weight<1.0: p1=_normalize_rows([self.market_model_weight*p1+(1-self.market_model_weight)*mk])[0]; used_market=True
        def binary(model,temp): raw=_temperature(model.predict_proba(x),temp)[0]; return self._ordered_probs(model,raw,[0,1])
        pg,pc,pt=binary(self.modelo_goles,self.temp_goles),binary(self.modelo_corners,self.temp_corners),binary(self.modelo_tarjetas,self.temp_tarjetas)
        return {"Resultado_1X2":{"Gana Local":round(p1[2]*100,1),"Empate":round(p1[1]*100,1),"Gana Visita":round(p1[0]*100,1)},"Goles_Over_Under":{"Over 2.5":round(pg[1]*100,1),"Under 2.5":round(pg[0]*100,1)},"Corners_Totales":{"Over 9.5 Corners":round(pc[1]*100,1),"Under 9.5 Corners":round(pc[0]*100,1)},"Tarjetas_Totales":{"Over 4.5 Tarjetas":round(pt[1]*100,1),"Under 4.5 Tarjetas":round(pt[0]*100,1)},"Meta":{"train_rows":self.n_train,"pregame_only":True,"temperature_1x2":self.temp_1x2,"market_model_weight":self.market_model_weight,"market_blend_used":used_market,"calibration_rows":self.calibration_rows,"rest_local":pl["rest_days"],"rest_visita":pv["rest_days"],"event_features":self.use_event_features,"event_games_local":pl.get("event_games",0),"event_games_visita":pv.get("event_games",0)}}
