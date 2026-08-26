"""Walk-forward temporal para las cinco ligas.

Cada bloque se predice con un modelo entrenado sólo con partidos anteriores.
Compara modelo puro, mercado no-vig y probabilidad final calibrada. No usa datos
posteriores para elegir temperatura ni peso modelo/mercado.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

from modules.elo_europa import SistemaEloEuropa
from modules.ml_europa import PredictorMLEuropa

FILES={"Premier":"data/historico_premier.csv","LaLiga":"data/historico_laliga.csv","SerieA":"data/historico_seriea.csv","Bundesliga":"data/historico_bundesliga.csv","Ligue1":"data/historico_ligue1.csv"}


def rating(tabla,team):
    try:return float(tabla.loc[tabla["Equipo"]==team,"ELO_Rating"].iloc[0])
    except Exception:return 1500.0


def market_vec(r):
    try:
        odds=np.array([float(r["Cuota_2"]),float(r["Cuota_X"]),float(r["Cuota_1"])])
        if np.any(~np.isfinite(odds)) or np.any(odds<=1.0):return None
        q=1.0/odds; return q/q.sum()
    except Exception:return None


def pred_vec(ml,loc,vis,el,ev,fecha,odds=None):
    p=ml.predecir_mercados_completos(loc,vis,elo_local=el,elo_visita=ev,cuotas_1x2=odds,fecha_partido=fecha)["Resultado_1X2"]
    v=np.array([p["Gana Visita"],p["Empate"],p["Gana Local"]],dtype=float)/100.0
    return v/v.sum()


def evaluate(path,n_blocks=3):
    df=pd.read_csv(path); df["_fecha"]=pd.to_datetime(df["Fecha"],errors="coerce",format="%Y-%m-%d")
    df=df.sort_values("_fecha",kind="stable").reset_index(drop=True)
    start=max(400,int(len(df)*0.70)); remaining=len(df)-start
    block=max(80,int(np.ceil(remaining/n_blocks)))
    y=[]; pure=[]; final=[]; market=[]; market_y=[]; learned=[]
    for bstart in range(start,len(df),block):
        bend=min(len(df),bstart+block); train=df.iloc[:bstart].copy(); test=df.iloc[bstart:bend].copy()
        if len(test)<20:continue
        ml=PredictorMLEuropa(); assert ml.entrenar(train),path
        tabla=SistemaEloEuropa().actualizar_ratings(train); learned.append((ml.market_model_weight,ml.temp_1x2))
        for _,r in test.iterrows():
            loc,vis=str(r["Local"]),str(r["Visitante"]); el,ev=rating(tabla,loc),rating(tabla,vis)
            gl,gv=float(r["Goles_Local"]),float(r["Goles_Visita"]); target=2 if gl>gv else (1 if gl==gv else 0)
            fecha=r.get("Fecha"); pv=pred_vec(ml,loc,vis,el,ev,fecha,None)
            odds={"1":r.get("Cuota_1"),"X":r.get("Cuota_X"),"2":r.get("Cuota_2")}; fv=pred_vec(ml,loc,vis,el,ev,fecha,odds)
            if not np.all(np.isfinite(pv)) or not np.all(np.isfinite(fv)):continue
            y.append(target); pure.append(pv); final.append(fv)
            mv=market_vec(r)
            if mv is not None:market.append(mv); market_y.append(target)
    y=np.asarray(y,dtype=int); pure=np.asarray(pure); final=np.asarray(final)
    assert len(y)>=100,(path,len(y)); onehot=np.eye(3)[y]
    out={
        "n":int(len(y)),"accuracy_pure":float(accuracy_score(y,pure.argmax(axis=1))),"accuracy_final":float(accuracy_score(y,final.argmax(axis=1))),
        "logloss_pure":float(log_loss(y,pure,labels=[0,1,2])),"logloss_final":float(log_loss(y,final,labels=[0,1,2])),
        "brier_final":float(np.mean(np.sum((final-onehot)**2,axis=1))),
        "mean_model_weight":float(np.mean([x[0] for x in learned])),"mean_temperature":float(np.mean([x[1] for x in learned])),"blocks":len(learned),
    }
    if len(market)>=80:
        out["market_n"]=len(market); out["logloss_market"]=float(log_loss(np.asarray(market_y),np.asarray(market),labels=[0,1,2]))
    return out


def main():
    out={name:evaluate(path) for name,path in FILES.items()}
    for name,m in out.items():print(name,{k:round(v,4) if isinstance(v,float) else v for k,v in m.items()})
    total=sum(v["n"] for v in out.values()); weights=[v["n"] for v in out.values()]
    pure_ll=float(np.average([v["logloss_pure"] for v in out.values()],weights=weights)); final_ll=float(np.average([v["logloss_final"] for v in out.values()],weights=weights))
    pure_acc=float(np.average([v["accuracy_pure"] for v in out.values()],weights=weights)); final_acc=float(np.average([v["accuracy_final"] for v in out.values()],weights=weights))
    assert total>=1000; assert np.isfinite(final_ll) and final_ll<1.099
    assert final_ll<=pure_ll+0.008,(pure_ll,final_ll)
    print("WALK_FORWARD_OK",{"n":total,"pure_logloss":round(pure_ll,4),"final_logloss":round(final_ll,4),"pure_accuracy":round(pure_acc,4),"final_accuracy":round(final_acc,4)})


if __name__=="__main__":main()
