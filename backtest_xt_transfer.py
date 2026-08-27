"""Compara baseline vs xThreat/PPDA transferidos en exactamente los mismos bloques OOS."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from modules.elo_europa import SistemaEloEuropa
from modules.ml_europa import PredictorMLEuropa

FILES={"Premier":"data/historico_premier.csv","LaLiga":"data/historico_laliga.csv","SerieA":"data/historico_seriea.csv","Bundesliga":"data/historico_bundesliga.csv","Ligue1":"data/historico_ligue1.csv"}


def rating(tabla,team):
    try: return float(tabla.loc[tabla['Equipo']==team,'ELO_Rating'].iloc[0])
    except Exception: return 1500.0


def odds(r):
    if {'Apertura_1','Apertura_X','Apertura_2'}.issubset(r.index): return {'1':r.get('Apertura_1'),'X':r.get('Apertura_X'),'2':r.get('Apertura_2')}
    return {'1':r.get('Cuota_1'),'X':r.get('Cuota_X'),'2':r.get('Cuota_2')}


def vec(model,r,tabla):
    loc,vis=str(r['Local']),str(r['Visitante'])
    p=model.predecir_mercados_completos(loc,vis,elo_local=rating(tabla,loc),elo_visita=rating(tabla,vis),cuotas_1x2=odds(r),fecha_partido=r.get('Fecha'))['Resultado_1X2']
    v=np.array([p['Gana Visita'],p['Empate'],p['Gana Local']],dtype=float)/100.0
    return v/v.sum()


def evaluate(path,n_blocks=3):
    df=pd.read_csv(path); df['_fecha']=pd.to_datetime(df['Fecha'],errors='coerce'); df=df.sort_values('_fecha',kind='stable').reset_index(drop=True)
    start=max(400,int(len(df)*0.70)); block=max(80,int(np.ceil((len(df)-start)/n_blocks)))
    y=[]; base=[]; event=[]; coverage=[]
    for bstart in range(start,len(df),block):
        bend=min(len(df),bstart+block); train=df.iloc[:bstart].copy(); test=df.iloc[bstart:bend].copy()
        if len(test)<20: continue
        mb=PredictorMLEuropa(False); me=PredictorMLEuropa(True)
        assert mb.entrenar(train) and me.entrenar(train),path
        tabla=SistemaEloEuropa().actualizar_ratings(train)
        for _,r in test.iterrows():
            gl,gv=float(r['Goles_Local']),float(r['Goles_Visita']); target=2 if gl>gv else (1 if gl==gv else 0)
            vb,ve=vec(mb,r,tabla),vec(me,r,tabla)
            if not np.all(np.isfinite(vb)) or not np.all(np.isfinite(ve)): continue
            y.append(target); base.append(vb); event.append(ve)
            coverage.append(int(me.raw_states.get(str(r['Local']),{}).get('event_games',0)>0 and me.raw_states.get(str(r['Visitante']),{}).get('event_games',0)>0))
    y=np.asarray(y,dtype=int); base=np.asarray(base); event=np.asarray(event); coverage=np.asarray(coverage,dtype=bool)
    assert len(y)>=100
    out={'n':len(y),'base_ll':log_loss(y,base,labels=[0,1,2]),'event_ll':log_loss(y,event,labels=[0,1,2]),'base_acc':accuracy_score(y,base.argmax(1)),'event_acc':accuracy_score(y,event.argmax(1)),'covered_n':int(coverage.sum())}
    if coverage.sum()>=50:
        out['covered_base_ll']=log_loss(y[coverage],base[coverage],labels=[0,1,2]); out['covered_event_ll']=log_loss(y[coverage],event[coverage],labels=[0,1,2])
    return out


def main():
    results={k:evaluate(v) for k,v in FILES.items()}; print('XT_TRANSFER_BY_LEAGUE',results)
    n=sum(r['n'] for r in results.values()); w=[r['n'] for r in results.values()]
    b=float(np.average([r['base_ll'] for r in results.values()],weights=w)); e=float(np.average([r['event_ll'] for r in results.values()],weights=w)); covered=sum(r['covered_n'] for r in results.values())
    summary={'n':n,'baseline_logloss':round(b,5),'event_logloss':round(e,5),'delta':round(e-b,5),'covered_n':covered}
    print('XT_TRANSFER_RESULT',summary)
    assert n>=1000
    assert np.isfinite(b) and np.isfinite(e)
    # Guardrail de seguridad: experimento puede empatar o empeorar levemente, pero no romperse.
    assert e <= b + 0.03,(b,e)

if __name__=='__main__': main()
