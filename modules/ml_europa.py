import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import log_loss

DEFAULT={"xg_for":1.25,"xg_against":1.25,"goals_for":1.30,"goals_against":1.30,"sot_for":4.0,"sot_against":4.0,"corners_for":4.8,"corners_against":4.8,"cards":2.0,"points":1.35,"games":0,"rest_days":7.0}
EMA_ALPHA=0.28
LINE_GRIDS={"goles":[1.5,2.0,2.5,3.0,3.5,4.0,4.5],"corners":[7.5,8.5,9.5,10.5,11.5,12.5],"tarjetas":[2.5,3.5,4.5,5.5,6.5,7.5]}

def _normalize_rows(p):
    p=np.asarray(p,dtype=float); p=np.clip(p,1e-6,1.0); return p/p.sum(axis=1,keepdims=True)
def _temperature(p,t):
    p=_normalize_rows(p); z=np.power(p,1.0/max(float(t),0.15)); return z/z.sum(axis=1,keepdims=True)
def _no_vig_1x2(h,d,a):
    try:
        odds=np.array([float(a),float(d),float(h)],dtype=float)
        if np.any(~np.isfinite(odds)) or np.any(odds<=1.0): return None
        q=1.0/odds; return q/q.sum()
    except Exception:return None

def _num(v):
    try:
        x=float(v); return x if np.isfinite(x) else np.nan
    except Exception:return np.nan

class PredictorMLEuropa:
    def __init__(self):
        # n_jobs=-1 removes the single-core cold-start bottleneck without reducing model quality.
        self.modelo_1x2=ExtraTreesClassifier(n_estimators=350,max_depth=10,min_samples_leaf=7,max_features=.80,random_state=42,class_weight="balanced",n_jobs=-1)
        self.modelo_goles=RandomForestClassifier(n_estimators=320,max_depth=10,min_samples_leaf=8,max_features=.85,random_state=43,class_weight="balanced_subsample",n_jobs=-1)
        self.modelo_corners=RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=9,max_features=.85,random_state=44,class_weight="balanced_subsample",n_jobs=-1)
        self.modelo_tarjetas=RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=9,max_features=.85,random_state=45,class_weight="balanced_subsample",n_jobs=-1)
        self.stats_equipos={}; self.raw_states={}; self.entrenado=False; self.n_train=0
        self.n_train_lines={"goles":0,"corners":0,"tarjetas":0}; self.temp_1x2=self.temp_goles=self.temp_corners=self.temp_tarjetas=1.0
        self.market_model_weight=1.0; self.calibration_rows=0
    @staticmethod
    def _safe(v,default):
        x=_num(v); return default if pd.isna(x) else x
    @staticmethod
    def _new_state():
        d={"games":0,"last_date":None}
        for k in ("xg_for","xg_against","goals_for","goals_against","sot_for","sot_against","corners_for","corners_against","cards","points"):
            d[k]=None; d[f"sum_{k}"]=0.0
        return d
    @staticmethod
    def _ema(old,new): return float(new) if old is None else EMA_ALPHA*float(new)+(1-EMA_ALPHA)*float(old)
    def _profile(self,state,match_date=None):
        if not state or int(state.get("games",0))<=0:return DEFAULT.copy()
        n=int(state["games"])
        def blend(k,d):
            long=state.get(f"sum_{k}",0)/n; recent=state.get(k)
            if recent is None or not np.isfinite(recent):recent=long
            return .60*float(recent)+.40*float(long if np.isfinite(long) else d)
        rest=7.0
        if match_date is not None and state.get("last_date") is not None:
            try:rest=float(np.clip((match_date-state["last_date"]).days,2,21))
            except Exception:pass
        return {"xg_for":blend("xg_for",1.25),"xg_against":blend("xg_against",1.25),"goals_for":blend("goals_for",1.30),"goals_against":blend("goals_against",1.30),"sot_for":blend("sot_for",4),"sot_against":blend("sot_against",4),"corners_for":blend("corners_for",4.8),"corners_against":blend("corners_against",4.8),"cards":blend("cards",2),"points":blend("points",1.35),"games":n,"rest_days":rest}
    @staticmethod
    def _elo_expected(a,b,home_adv=55.0):return 1/(1+10**((b-(a+home_adv))/400))
    def _features(self,pl,pv,elo_l,elo_v):
        al=.55*pl["xg_for"]+.25*pl["goals_for"]+.20*(pl["sot_for"]/3.2); av=.55*pv["xg_for"]+.25*pv["goals_for"]+.20*(pv["sot_for"]/3.2)
        dl=.60*pl["xg_against"]+.25*pl["goals_against"]+.15*(pl["sot_against"]/3.2); dv=.60*pv["xg_against"]+.25*pv["goals_against"]+.15*(pv["sot_against"]/3.2)
        el=max(.20,.58*al+.42*dv); ev=max(.20,.58*av+.42*dl)
        return [pl["xg_for"],pv["xg_for"],pl["xg_against"],pv["xg_against"],pl["goals_for"],pv["goals_for"],pl["goals_against"],pv["goals_against"],pl["sot_for"],pv["sot_for"],pl["sot_against"],pv["sot_against"],pl["corners_for"],pv["corners_for"],pl["corners_against"],pv["corners_against"],pl["cards"],pv["cards"],pl["points"],pv["points"],pl["rest_days"],pv["rest_days"],pl["rest_days"]-pv["rest_days"],el,ev,el-ev,(float(elo_l)+55-float(elo_v))/400,self._elo_expected(float(elo_l),float(elo_v)),np.log1p(pl["games"]),np.log1p(pv["games"])]
    @staticmethod
    def _line_features(base,line):line=float(line);return list(base)+[line,line*line,np.log1p(max(line,0))]
    @staticmethod
    def _best_temperature(model,X,y):
        if len(X)<80:return 1.0
        raw=model.predict_proba(X); classes=list(model.classes_); best=(1.0,float("inf"))
        for t in [.70,.85,1,1.15,1.30,1.50,1.75,2.0]:
            try:l=log_loss(y,_temperature(raw,t),labels=classes)
            except Exception:continue
            if l<best[1]:best=(t,l)
        return float(best[0])
    def _fit_with_temporal_calibration(self,model,X,y):
        n=len(X); cut=max(100,int(n*.80)); cut=min(cut,n-80) if n>=220 else n
        if cut<n:
            model.fit(X[:cut],y[:cut]); t=self._best_temperature(model,X[cut:],y[cut:]); self.calibration_rows=max(self.calibration_rows,n-cut)
        else:t=1.0
        model.fit(X,y); return t
    def entrenar(self,df_historico):
        self.entrenado=False; self.stats_equipos={}; self.raw_states={}; self.n_train_lines={"goles":0,"corners":0,"tarjetas":0}
        if df_historico is None or df_historico.empty:return False
        df=df_historico.copy(); df["_fecha"]=pd.to_datetime(df.get("Fecha"),errors="coerce",format="%Y-%m-%d"); df=df.sort_values("_fecha",kind="stable")
        states={};ratings={};X1=[];y1=[];markets=[];Xg=[];yg=[];Xc=[];yc=[];Xt=[];yt=[]
        for _,r in df.iterrows():
            loc,vis=r.get("Local"),r.get("Visitante")
            if pd.isna(loc) or pd.isna(vis):continue
            loc,vis=str(loc).strip(),str(vis).strip(); date=r.get("_fecha"); sl=states.setdefault(loc,self._new_state());sv=states.setdefault(vis,self._new_state());pl,pv=self._profile(sl,date),self._profile(sv,date);elo_l,elo_v=ratings.get(loc,1500.),ratings.get(vis,1500.)
            gl,gv=_num(r.get("Goles_Local")),_num(r.get("Goles_Visita"))
            if pd.isna(gl) or pd.isna(gv):continue
            cl_raw,cv_raw=_num(r.get("Corners_Local")),_num(r.get("Corners_Visita")); tl_raw,tv_raw=_num(r.get("Tarjetas_Local")),_num(r.get("Tarjetas_Visita")); goals=gl+gv
            if pl["games"]>=4 and pv["games"]>=4:
                base=self._features(pl,pv,elo_l,elo_v);X1.append(base);y1.append(2 if gl>gv else (1 if gl==gv else 0));markets.append(_no_vig_1x2(r.get("Cuota_1"),r.get("Cuota_X"),r.get("Cuota_2")))
                for line in LINE_GRIDS["goles"]:
                    if abs(goals-line)>=1e-9:Xg.append(self._line_features(base,line));yg.append(int(goals>line))
                # Missing observations are skipped, never converted into false zero/Under labels.
                if np.isfinite(cl_raw) and np.isfinite(cv_raw):
                    total=cl_raw+cv_raw
                    for line in LINE_GRIDS["corners"]:
                        if abs(total-line)>=1e-9:Xc.append(self._line_features(base,line));yc.append(int(total>line))
                if np.isfinite(tl_raw) and np.isfinite(tv_raw):
                    total=tl_raw+tv_raw
                    for line in LINE_GRIDS["tarjetas"]:
                        if abs(total-line)>=1e-9:Xt.append(self._line_features(base,line));yt.append(int(total>line))
            xgl,xgv=self._safe(r.get("xG_Local"),gl),self._safe(r.get("xG_Visita"),gv);sotl,sotv=self._safe(r.get("TirosGol_Local"),4),self._safe(r.get("TirosGol_Visita"),4);cl,cv=self._safe(r.get("Corners_Local"),4.8),self._safe(r.get("Corners_Visita"),4.8);cal,cav=self._safe(r.get("Tarjetas_Local"),2),self._safe(r.get("Tarjetas_Visita"),2);pts_l=3 if gl>gv else (1 if gl==gv else 0);pts_v=3 if gv>gl else (1 if gl==gv else 0)
            for st,xgf,xga,gf,ga,sf,sa,cf,ca,cards,pts in [(sl,xgl,xgv,gl,gv,sotl,sotv,cl,cv,cal,pts_l),(sv,xgv,xgl,gv,gl,sotv,sotl,cv,cl,cav,pts_v)]:
                vals={"xg_for":xgf,"xg_against":xga,"goals_for":gf,"goals_against":ga,"sot_for":sf,"sot_against":sa,"corners_for":cf,"corners_against":ca,"cards":cards,"points":pts};st["games"]+=1
                for k,v in vals.items():v=float(v);st[k]=self._ema(st.get(k),v);st[f"sum_{k}"]+=v
                if pd.notna(date):st["last_date"]=date
            exp=self._elo_expected(elo_l,elo_v);score=1 if gl>gv else (.5 if gl==gv else 0);k=22*(1+.12*min(abs(gl-gv),4));delta=k*(score-exp);ratings[loc],ratings[vis]=elo_l+delta,elo_v-delta
        if len(X1)<250 or len(set(y1))<3 or min(len(Xg),len(Xc),len(Xt))<500:return False
        X1,y1=np.asarray(X1,float),np.asarray(y1);Xg,yg=np.asarray(Xg,float),np.asarray(yg);Xc,yc=np.asarray(Xc,float),np.asarray(yc);Xt,yt=np.asarray(Xt,float),np.asarray(yt)
        self.temp_1x2=self._fit_with_temporal_calibration(self.modelo_1x2,X1,y1);self.temp_goles=self._fit_with_temporal_calibration(self.modelo_goles,Xg,yg);self.temp_corners=self._fit_with_temporal_calibration(self.modelo_corners,Xc,yc);self.temp_tarjetas=self._fit_with_temporal_calibration(self.modelo_tarjetas,Xt,yt)
        cut=int(len(X1)*.80)
        if len(X1)-cut>=80:
            tmp=ExtraTreesClassifier(n_estimators=280,max_depth=10,min_samples_leaf=7,max_features=.80,random_state=142,class_weight="balanced",n_jobs=-1);tmp.fit(X1[:cut],y1[:cut]);pm=_temperature(tmp.predict_proba(X1[cut:]),self.temp_1x2);idx=[];marr=[];target=[]
            for j,mk in enumerate(markets[cut:]):
                if mk is not None:idx.append(j);marr.append(mk);target.append(y1[cut+j])
            if len(idx)>=60:
                pmv,marr,target=pm[np.asarray(idx)],np.asarray(marr),np.asarray(target);best=(1.,float("inf"))
                for w in [.25,.40,.55,.70,.85,1.0]:
                    l=log_loss(target,_normalize_rows(w*pmv+(1-w)*marr),labels=[0,1,2])
                    if l<best[1]:best=(w,l)
                self.market_model_weight=float(best[0])
        self.raw_states=states;self.stats_equipos={t:self._profile(s,None) for t,s in states.items()};self.n_train=len(X1);self.n_train_lines={"goles":len(Xg),"corners":len(Xc),"tarjetas":len(Xt)};self.entrenado=True;return True
    @staticmethod
    def _ordered_probs(model,raw,wanted):
        classes=list(model.classes_);return np.asarray([float(raw[classes.index(c)]) if c in classes else 0 for c in wanted])
    def _predict_binary_line(self,model,temp,base,line):
        raw=_temperature(model.predict_proba(np.asarray([self._line_features(base,line)],float)),temp)[0];return self._ordered_probs(model,raw,[0,1])
    def predecir_mercados_completos(self,local,visita,goles_sim_l=None,goles_sim_v=None,elo_local=1500,elo_visita=1500,cuotas_1x2=None,fecha_partido=None,lineas_casino=None):
        if not self.entrenado:return {"error":"El modelo no está entrenado (faltan datos históricos)."}
        date=pd.to_datetime(fecha_partido,errors="coerce") if fecha_partido else None
        if date is not None and pd.isna(date):date=None
        pl=self._profile(self.raw_states.get(local),date) if local in self.raw_states else DEFAULT;pv=self._profile(self.raw_states.get(visita),date) if visita in self.raw_states else DEFAULT;base=self._features(pl,pv,elo_local,elo_visita);raw1=_temperature(self.modelo_1x2.predict_proba(np.asarray([base],float)),self.temp_1x2)[0];p1=self._ordered_probs(self.modelo_1x2,raw1,[0,1,2]);used=False
        if cuotas_1x2:
            mk=_no_vig_1x2(cuotas_1x2.get("1"),cuotas_1x2.get("X"),cuotas_1x2.get("2"))
            if mk is not None and self.market_model_weight<1:p1=_normalize_rows([self.market_model_weight*p1+(1-self.market_model_weight)*mk])[0];used=True
        lineas={"goles":2.5,"corners":9.5,"tarjetas":4.5};source=lineas_casino
        if not isinstance(source,dict) and isinstance(cuotas_1x2,dict):source=cuotas_1x2.get("_lineas",{})
        if isinstance(source,dict):
            for k in lineas:
                try:
                    if source.get(k) is not None:lineas[k]=float(source[k])
                except Exception:pass
        pg=self._predict_binary_line(self.modelo_goles,self.temp_goles,base,lineas["goles"]);pc=self._predict_binary_line(self.modelo_corners,self.temp_corners,base,lineas["corners"]);pt=self._predict_binary_line(self.modelo_tarjetas,self.temp_tarjetas,base,lineas["tarjetas"]);gl,cl,tl=lineas["goles"],lineas["corners"],lineas["tarjetas"]
        return {"Resultado_1X2":{"Gana Local":round(p1[2]*100,1),"Empate":round(p1[1]*100,1),"Gana Visita":round(p1[0]*100,1)},"Goles_Over_Under":{f"Over {gl:g}":round(pg[1]*100,1),f"Under {gl:g}":round(pg[0]*100,1)},"Corners_Totales":{f"Over {cl:g} Corners":round(pc[1]*100,1),f"Under {cl:g} Corners":round(pc[0]*100,1)},"Tarjetas_Totales":{f"Over {tl:g} Tarjetas":round(pt[1]*100,1),f"Under {tl:g} Tarjetas":round(pt[0]*100,1)},"Meta":{"train_rows":self.n_train,"train_rows_line_aware":self.n_train_lines,"pregame_only":True,"line_aware":True,"lineas_modeladas":lineas,"temperature_1x2":self.temp_1x2,"temperature_goles":self.temp_goles,"temperature_corners":self.temp_corners,"temperature_tarjetas":self.temp_tarjetas,"market_model_weight":self.market_model_weight,"market_blend_used":used,"calibration_rows":self.calibration_rows,"rest_local":pl["rest_days"],"rest_visita":pv["rest_days"],"parallel_training":True,"missing_market_targets_skipped":True}}
