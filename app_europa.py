import os
import difflib
import streamlit as st
import pandas as pd
import requests

from modules.elo_europa import SistemaEloEuropa
from modules.montecarlo_europa import simular_partido_europa
from modules.ml_europa import PredictorMLEuropa
from modules.odds_europa import obtener_cuotas_europa, analizar_apuestas_europa
from modules.data_store import cargar_historico

st.set_page_config(page_title="European Elite Leagues Analytics", layout="wide", page_icon="⚽")

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY} if API_KEY else {}
LIGAS_IDS = {"Premier League":39,"La Liga":140,"Serie A":135,"Bundesliga":78,"Ligue 1":61}
ESPN_LIGAS_MAP = {39:"eng.1",140:"esp.1",135:"ita.1",78:"ger.1",61:"fra.1"}
ARCHIVOS_HISTORICOS = {"Premier League":"data/historico_premier.csv","La Liga":"data/historico_laliga.csv","Serie A":"data/historico_seriea.csv","Bundesliga":"data/historico_bundesliga.csv","Ligue 1":"data/historico_ligue1.csv"}

def _norm(s): return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def resolver_nombre(nombre, df):
    equipos=sorted(set(df.get("Local",pd.Series(dtype=str)).dropna()).union(set(df.get("Visitante",pd.Series(dtype=str)).dropna())))
    if not equipos: return nombre
    target=_norm(nombre); exact=[e for e in equipos if _norm(e)==target]
    if exact: return exact[0]
    scored=[(difflib.SequenceMatcher(None,target,_norm(e)).ratio(),e) for e in equipos]
    score,best=max(scored,default=(0,nombre)); return best if score>=0.72 else nombre

@st.cache_data(ttl=1800)
def cargar_historico_liga(nombre_liga):
    try:
        df=cargar_historico(ARCHIVOS_HISTORICOS[nombre_liga])
        if df.empty: return df
        df["Local"]=df["Local"].astype(str).str.strip(); df["Visitante"]=df["Visitante"].astype(str).str.strip(); return df
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=600)
def obtener_proximos_partidos_europa(league_id):
    partidos={}
    if API_KEY:
        try:
            r=requests.get(f"{BASE_URL}/fixtures",headers=HEADERS,params={"league":league_id,"season":2026,"next":15},timeout=8)
            if r.status_code==200:
                for p in r.json().get("response",[]):
                    local=p.get("teams",{}).get("home",{}).get("name"); visita=p.get("teams",{}).get("away",{}).get("name")
                    fecha=str(p.get("fixture",{}).get("date",""))[:10]; fid=p.get("fixture",{}).get("id")
                    if local and visita: partidos[f"⚽ {fecha} | {local} vs {visita}"]={"local":local,"visita":visita,"fixture_id":fid,"fecha":fecha}
        except Exception: pass
    if not partidos and league_id in ESPN_LIGAS_MAP:
        try:
            url=f"https://site.api.espn.com/apis/site/v2/sports/soccer/{ESPN_LIGAS_MAP[league_id]}/scoreboard"; r=requests.get(url,timeout=8)
            if r.status_code==200:
                for event in r.json().get("events",[]):
                    comp=event.get("competitions",[{}])[0]; competitors=comp.get("competitors",[])
                    local=next((x.get("team",{}).get("displayName") for x in competitors if x.get("homeAway")=="home"),None)
                    visita=next((x.get("team",{}).get("displayName") for x in competitors if x.get("homeAway")=="away"),None)
                    fecha=str(event.get("date",""))[:10]
                    if local and visita: partidos[f"⚽ {fecha} | {local} vs {visita}"]={"local":local,"visita":visita,"fixture_id":999999,"fecha":fecha}
        except Exception: pass
    return partidos

@st.cache_resource(show_spinner=False)
def construir_motores_cache(nombre_liga, data_version):
    df=cargar_historico_liga(nombre_liga)
    elo=SistemaEloEuropa(); tabla=elo.actualizar_ratings(df); ml=PredictorMLEuropa(); ml_ok=ml.entrenar(df)
    return tabla,ml,ml_ok

def obtener_motores(nombre_liga,df):
    try: version=f"{len(df)}:{str(df.iloc[-1].get('Fecha',''))}"
    except Exception: version=str(len(df))
    return construir_motores_cache(nombre_liga,version)

def rating(tabla,team):
    try: return float(tabla.loc[tabla["Equipo"]==team,"ELO_Rating"].iloc[0])
    except Exception: return 1500.0

def analizar_partido(nombre_liga,fixture):
    df=cargar_historico_liga(nombre_liga)
    if df.empty: return None,None,None,"Sin histórico disponible"
    loc_api,vis_api=fixture["local"],fixture["visita"]; loc,vis=resolver_nombre(loc_api,df),resolver_nombre(vis_api,df)
    tabla,ml,ml_ok=obtener_motores(nombre_liga,df); e_loc,e_vis=rating(tabla,loc),rating(tabla,vis)
    odds=obtener_cuotas_europa(fixture.get("fixture_id"),nombre_liga,loc_api,vis_api)
    lineas_casino=odds.get("_lineas",{}) if isinstance(odds,dict) else {}
    mc=simular_partido_europa(loc,vis,df,e_loc,e_vis,lineas_casino=lineas_casino)
    preds=ml.predecir_mercados_completos(loc,vis,elo_local=e_loc,elo_visita=e_vis,cuotas_1x2=odds,fecha_partido=fixture.get("fecha")) if ml_ok else {}
    bets=analizar_apuestas_europa(mc,preds,fixture.get("fixture_id"),cuotas_personalizadas=odds,nombre_liga=nombre_liga,local=loc_api,visita=vis_api)
    mm=preds.get("Meta",{}) if isinstance(preds,dict) else {}
    meta={"local_modelo":loc,"visita_modelo":vis,"elo_local":round(e_loc,1),"elo_visita":round(e_vis,1),"ml_ok":ml_ok,"train_rows":getattr(ml,"n_train",0),"temperature_1x2":mm.get("temperature_1x2",1.0),"market_model_weight":mm.get("market_model_weight",1.0),"market_blend_used":mm.get("market_blend_used",False),"rest_local":mm.get("rest_local",7.0),"rest_visita":mm.get("rest_visita",7.0),"lineas_casino":lineas_casino,"bookmaker":odds.get("_bookmaker") if isinstance(odds,dict) else None}
    return mc,preds,bets,meta

st.title("🇪🇺 European Elite Leagues Analytics")
st.caption("Parquet + DuckDB · fixtures reales · forma reciente y descanso real · ML temporal calibrado · mercado no-vig aprendido · Monte Carlo sobre línea real del casino · EV sólo con cuota real.")
with st.expander("✅ Estado del sistema",expanded=True):
    cols=st.columns(5)
    for i,liga in enumerate(LIGAS_IDS):
        df=cargar_historico_liga(liga); cols[i].metric(liga,f"{len(df):,} partidos","OK" if len(df)>=150 else "DATOS INSUFICIENTES")
    st.caption("La app prefiere Parquet consultado con DuckDB y conserva CSV como respaldo. API-Sports es opcional; ESPN sirve de respaldo para fixtures.")

labels=["🇬🇧 Premier League","🇪🇸 La Liga","🇮🇹 Serie A","🇩🇪 Bundesliga","🇫🇷 Ligue 1","🌐 Escáner Global EV+"]; tabs=st.tabs(labels)
for idx,(liga,league_id) in enumerate(LIGAS_IDS.items()):
    with tabs[idx]:
        st.subheader(f"📊 {liga}"); fixtures=obtener_proximos_partidos_europa(league_id)
        if not fixtures: st.warning("No hay fixtures próximos disponibles en este momento."); continue
        key=st.selectbox("Próximo partido",list(fixtures.keys()),key=f"sel_{league_id}"); fx=fixtures[key]
        if st.button("Ejecutar análisis",key=f"btn_{league_id}",type="primary"):
            with st.spinner("Ejecutando motores prepartido..."): mc,ml,bets,meta=analizar_partido(liga,fx)
            if mc is None: st.error(meta); continue
            st.caption(f"Mapeo: {meta['local_modelo']} vs {meta['visita_modelo']} · Elo {meta['elo_local']} / {meta['elo_visita']} · ML n={meta['train_rows']} · Temp={meta['temperature_1x2']} · peso modelo={meta['market_model_weight']:.2f} · descanso {meta['rest_local']:.0f}/{meta['rest_visita']:.0f} días")
            if meta.get("lineas_casino"):
                st.success(f"🎰 Línea casino usada por Monte Carlo: {meta['lineas_casino']}" + (f" · {meta['bookmaker']}" if meta.get('bookmaker') else ""))
            else:
                st.warning("No se recibió línea O/U real del casino para este fixture; no se fabricará una línea para apostar.")
            c1,c2,c3=st.columns(3); c1.metric("Local MC",f"{mc['Resultado_1X2']['Gana Local']}%"); c2.metric("Empate MC",f"{mc['Resultado_1X2']['Empate']}%"); c3.metric("Visita MC",f"{mc['Resultado_1X2']['Gana Visita']}%")
            if ml and "Resultado_1X2" in ml:
                m1,m2,m3=st.columns(3); m1.metric("Local ML calibrado",f"{ml['Resultado_1X2']['Gana Local']}%"); m2.metric("Empate ML calibrado",f"{ml['Resultado_1X2']['Empate']}%"); m3.metric("Visita ML calibrado",f"{ml['Resultado_1X2']['Gana Visita']}%")
            st.markdown("#### Mercados y valor")
            if bets is None or bets.empty: st.info("No hay cuotas reales suficientes para calcular oportunidades EV+.")
            else: st.dataframe(bets,use_container_width=True,hide_index=True)

with tabs[5]:
    st.subheader("🌐 Escáner Global EV+")
    st.info("Analiza exclusivamente fixtures reales y exige cuota real. Monte Carlo usa la línea O/U publicada por el bookmaker; ML sólo valida una línea si coincide con la línea para la que fue entrenado.")
    if st.button("🚀 Escanear próximas jornadas",type="primary",key="scan_global"):
        status=st.status("Iniciando escáner global...",expanded=True)
        rows=[]; errores=[]; total_fixtures=0; analizados=0
        progress=st.progress(0)
        for i,(liga,league_id) in enumerate(LIGAS_IDS.items()):
            fixtures=obtener_proximos_partidos_europa(league_id)
            total_fixtures += len(fixtures)
            status.write(f"{liga}: {len(fixtures)} fixtures encontrados")
            if not fixtures:
                errores.append(f"{liga}: sin fixtures disponibles")
            for label,fx in fixtures.items():
                analizados += 1
                status.update(label=f"Analizando {liga}: {fx['local']} vs {fx['visita']}")
                try:
                    mc,ml,bets,meta=analizar_partido(liga,fx)
                    if mc is None:
                        errores.append(f"{liga} · {fx['local']} vs {fx['visita']}: {meta}")
                        continue
                    if bets is None or bets.empty: continue
                    good=bets[bets["Veredicto"].astype(str).str.contains("🔥|✅",regex=True,na=False)].copy()
                    if good.empty: continue
                    good.insert(0,"Liga",liga); good.insert(1,"Partido",f"{fx['local']} vs {fx['visita']}"); good.insert(2,"Fecha",fx.get("fecha","")); rows.append(good)
                except Exception as exc:
                    errores.append(f"{liga} · {fx.get('local','?')} vs {fx.get('visita','?')}: {type(exc).__name__}: {exc}")
            progress.progress(int((i+1)/len(LIGAS_IDS)*100))
        progress.empty()
        status.update(label=f"Escaneo terminado: {analizados}/{total_fixtures} fixtures procesados",state="complete",expanded=False)
        if rows:
            out=pd.concat(rows,ignore_index=True); st.success(f"Se encontraron {len(out)} oportunidades que pasan los filtros conservadores."); st.dataframe(out,use_container_width=True,hide_index=True)
        elif total_fixtures == 0:
            st.warning("No se encontraron fixtures en ninguna liga. Revisa conectividad/API-Sports; ESPN puede no publicar jornadas futuras con suficiente anticipación.")
        else:
            st.info(f"Se analizaron {analizados} fixtures y ninguno pasó los filtros EV+. NO BET es un resultado válido.")
        if errores:
            with st.expander(f"Diagnóstico del escáner ({len(errores)} avisos)"):
                for err in errores[:50]: st.write("•",err)
