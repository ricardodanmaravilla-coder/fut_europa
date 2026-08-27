"""Une StatsBomb Open Data con históricos Football-Data.

Prioriza coincidencia exacta; si los nombres difieren, permite fuzzy matching sólo en
la misma fecha, mismo orden local/visita y score conjunto >= 0.88. Guarda score de
match para auditoría. Sustituye xG proxy únicamente en matches validados.
"""
import difflib
import glob
import os
import unicodedata
import numpy as np
import pandas as pd

SB='data/statsbomb_xg_matches.csv'
ADV_COLS=['Pases_SB_Local','Pases_SB_Visita','Presiones_SB_Local','Presiones_SB_Visita','Posesiones_SB_Local','Posesiones_SB_Visita','PPDA_Local','PPDA_Visita','xThreat_Local','xThreat_Visita']


def norm(x):
    s=unicodedata.normalize('NFKD',str(x or '')).encode('ascii','ignore').decode().lower()
    aliases={'manchesterunited':'manunited','manchestercity':'mancity','parissaintgermain':'psg','internazionale':'inter','intermilan':'inter','bayernmunich':'bayernmunchen','borussiamonchengladbach':'monchengladbach'}
    k=''.join(c for c in s if c.isalnum())
    return aliases.get(k,k)


def sim(a,b): return difflib.SequenceMatcher(None,norm(a),norm(b)).ratio()


def main():
    if not os.path.exists(SB):
        print('MERGE_SB_SKIP no cache'); return
    sb=pd.read_csv(SB); sb['Fecha']=sb['Fecha'].astype(str)
    total=exact=fuzzy=0
    for path in glob.glob('data/historico_*.csv'):
        if 'statsbomb' in path: continue
        df=pd.read_csv(path); df['Fecha']=df['Fecha'].astype(str)
        for c in ADV_COLS+['StatsBomb_match_score']:
            if c not in df.columns: df[c]=np.nan
        bydate={d:g for d,g in sb.groupby('Fecha')}
        matched=0
        for i,r in df.iterrows():
            cand=bydate.get(str(r['Fecha']))
            if cand is None: continue
            best=None; bestscore=0.0; isexact=False
            for _,s in cand.iterrows():
                sl=sim(r['Local'],s['Local_SB']); sv=sim(r['Visitante'],s['Visitante_SB']); score=(sl+sv)/2
                ex=norm(r['Local'])==norm(s['Local_SB']) and norm(r['Visitante'])==norm(s['Visitante_SB'])
                if ex: best=s; bestscore=1.0; isexact=True; break
                if sl>=0.84 and sv>=0.84 and score>bestscore: best=s; bestscore=score
            if best is None or (not isexact and bestscore<0.88): continue
            matched+=1; exact+=int(isexact); fuzzy+=int(not isexact)
            df.at[i,'StatsBomb_match_score']=round(bestscore,4)
            for c in ADV_COLS:
                if c in best.index: df.at[i,c]=best.get(c)
            if pd.notna(best.get('xG_Real_Local')) and pd.notna(best.get('xG_Real_Visita')):
                df.at[i,'xG_Local']=best['xG_Real_Local']; df.at[i,'xG_Visita']=best['xG_Real_Visita']; df.at[i,'Fuente_xG']='StatsBomb Open Data'
                df.at[i,'StatsBomb_match_id']=best.get('StatsBomb_match_id')
        df.to_csv(path,index=False); total+=matched
        print('MERGE_SB',path,'matched',matched)
    print('MERGE_SB_OK',{'matched':total,'exact':exact,'fuzzy':fuzzy})

if __name__=='__main__': main()
