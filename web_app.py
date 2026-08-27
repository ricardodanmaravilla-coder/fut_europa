import os
import time
import difflib
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from modules.elo_europa import SistemaEloEuropa
from modules.montecarlo_europa import simular_partido_europa
from modules.ml_europa import PredictorMLEuropa
from modules.odds_europa import obtener_cuotas_europa, analizar_apuestas_europa
from modules.data_store import cargar_historico

app = FastAPI(title="FUT Europa", version="2.0")

API_KEY = os.environ.get("API_SPORTS_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY} if API_KEY else {}
LIGAS_IDS = {"Premier League":39,"La Liga":140,"Serie A":135,"Bundesliga":78,"Ligue 1":61}
ESPN_LIGAS_MAP = {39:"eng.1",140:"esp.1",135:"ita.1",78:"ger.1",61:"fra.1"}
ARCHIVOS_HISTORICOS = {"Premier League":"data/historico_premier.csv","La Liga":"data/historico_laliga.csv","Serie A":"data/historico_seriea.csv","Bundesliga":"data/historico_bundesliga.csv","Ligue 1":"data/historico_ligue1.csv"}

_fixture_cache: dict[int, tuple[float, dict[str, dict[str, Any]]]] = {}


def _norm(s):
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def resolver_nombre(nombre, df):
    equipos = sorted(set(df.get("Local", pd.Series(dtype=str)).dropna()).union(set(df.get("Visitante", pd.Series(dtype=str)).dropna())))
    if not equipos:
        return nombre
    target = _norm(nombre)
    exact = [e for e in equipos if _norm(e) == target]
    if exact:
        return exact[0]
    scored = [(difflib.SequenceMatcher(None, target, _norm(e)).ratio(), e) for e in equipos]
    score, best = max(scored, default=(0, nombre))
    return best if score >= 0.72 else nombre


@lru_cache(maxsize=8)
def cargar_historico_liga(nombre_liga: str):
    try:
        df = cargar_historico(ARCHIVOS_HISTORICOS[nombre_liga])
        if df.empty:
            return df
        df = df.copy()
        df["Local"] = df["Local"].astype(str).str.strip()
        df["Visitante"] = df["Visitante"].astype(str).str.strip()
        return df
    except Exception:
        return pd.DataFrame()


def obtener_proximos_partidos_europa(league_id: int):
    now = time.time()
    cached = _fixture_cache.get(league_id)
    if cached and now - cached[0] < 300:
        return cached[1]
    partidos = {}
    if API_KEY:
        try:
            r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params={"league": league_id, "season": 2026, "next": 15}, timeout=8)
            if r.status_code == 200:
                for p in r.json().get("response", []):
                    local = p.get("teams", {}).get("home", {}).get("name")
                    visita = p.get("teams", {}).get("away", {}).get("name")
                    fecha = str(p.get("fixture", {}).get("date", ""))[:10]
                    fid = p.get("fixture", {}).get("id")
                    if local and visita:
                        partidos[str(fid or f"{fecha}-{local}-{visita}")] = {"local": local, "visita": visita, "fixture_id": fid, "fecha": fecha}
        except Exception:
            pass
    if not partidos and league_id in ESPN_LIGAS_MAP:
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{ESPN_LIGAS_MAP[league_id]}/scoreboard"
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                for event in r.json().get("events", []):
                    comp = event.get("competitions", [{}])[0]
                    competitors = comp.get("competitors", [])
                    local = next((x.get("team", {}).get("displayName") for x in competitors if x.get("homeAway") == "home"), None)
                    visita = next((x.get("team", {}).get("displayName") for x in competitors if x.get("homeAway") == "away"), None)
                    fecha = str(event.get("date", ""))[:10]
                    eid = str(event.get("id", f"{fecha}-{local}-{visita}"))
                    if local and visita:
                        partidos[eid] = {"local": local, "visita": visita, "fixture_id": 999999, "fecha": fecha}
        except Exception:
            pass
    _fixture_cache[league_id] = (now, partidos)
    return partidos


@lru_cache(maxsize=8)
def construir_motores_cache(nombre_liga: str, data_version: str):
    df = cargar_historico_liga(nombre_liga)
    elo = SistemaEloEuropa()
    tabla = elo.actualizar_ratings(df)
    ml = PredictorMLEuropa()
    ml_ok = ml.entrenar(df)
    return tabla, ml, ml_ok


def obtener_motores(nombre_liga, df):
    try:
        version = f"{len(df)}:{str(df.iloc[-1].get('Fecha',''))}"
    except Exception:
        version = str(len(df))
    return construir_motores_cache(nombre_liga, version)


def rating(tabla, team):
    try:
        return float(tabla.loc[tabla["Equipo"] == team, "ELO_Rating"].iloc[0])
    except Exception:
        return 1500.0


def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if pd.isna(v) if not isinstance(v, (str, bytes, dict, list, tuple)) else False:
        return None
    return v


def analizar_partido(nombre_liga, fixture):
    df = cargar_historico_liga(nombre_liga)
    if df.empty:
        raise ValueError("Sin histórico disponible")
    loc_api, vis_api = fixture["local"], fixture["visita"]
    loc, vis = resolver_nombre(loc_api, df), resolver_nombre(vis_api, df)
    tabla, ml, ml_ok = obtener_motores(nombre_liga, df)
    e_loc, e_vis = rating(tabla, loc), rating(tabla, vis)
    odds = obtener_cuotas_europa(fixture.get("fixture_id"), nombre_liga, loc_api, vis_api)
    mc = simular_partido_europa(loc, vis, df, e_loc, e_vis)
    preds = ml.predecir_mercados_completos(loc, vis, elo_local=e_loc, elo_visita=e_vis, cuotas_1x2=odds, fecha_partido=fixture.get("fecha")) if ml_ok else {}
    bets = analizar_apuestas_europa(mc, preds, fixture.get("fixture_id"), cuotas_personalizadas=odds, nombre_liga=nombre_liga, local=loc_api, visita=vis_api)
    mm = preds.get("Meta", {}) if isinstance(preds, dict) else {}
    meta = {"local_modelo": loc, "visita_modelo": vis, "elo_local": round(e_loc,1), "elo_visita": round(e_vis,1), "ml_ok": ml_ok, "train_rows": getattr(ml,"n_train",0), "temperature_1x2": mm.get("temperature_1x2",1.0), "market_model_weight": mm.get("market_model_weight",1.0), "market_blend_used": mm.get("market_blend_used",False), "rest_local": mm.get("rest_local",7.0), "rest_visita": mm.get("rest_visita",7.0)}
    bet_rows = [] if bets is None or bets.empty else bets.replace({np.nan: None}).to_dict(orient="records")
    return _clean({"mc": mc, "ml": preds, "bets": bet_rows, "meta": meta})


class AnalyzeRequest(BaseModel):
    liga: str
    fixture_key: str


@app.get("/health")
def health():
    return {"status": "ok", "runtime": "fastapi", "streamlit": False}


@app.get("/api/status")
def status():
    return {liga: {"partidos": len(cargar_historico_liga(liga)), "ready": len(cargar_historico_liga(liga)) >= 150} for liga in LIGAS_IDS}


@app.get("/api/fixtures/{liga}")
def fixtures(liga: str):
    if liga not in LIGAS_IDS:
        raise HTTPException(404, "Liga no válida")
    data = obtener_proximos_partidos_europa(LIGAS_IDS[liga])
    return [{"key": k, **v} for k, v in data.items()]


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if req.liga not in LIGAS_IDS:
        raise HTTPException(404, "Liga no válida")
    fixtures_map = obtener_proximos_partidos_europa(LIGAS_IDS[req.liga])
    fixture = fixtures_map.get(req.fixture_key)
    if not fixture:
        raise HTTPException(404, "Fixture no encontrado")
    try:
        return analizar_partido(req.liga, fixture)
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc


@app.get("/api/scan")
def scan():
    rows = []
    errores = []
    for liga, league_id in LIGAS_IDS.items():
        for _, fx in obtener_proximos_partidos_europa(league_id).items():
            try:
                result = analizar_partido(liga, fx)
                good = [r for r in result["bets"] if "🔥" in str(r.get("Veredicto", "")) or "✅" in str(r.get("Veredicto", ""))]
                for row in good:
                    rows.append({"Liga": liga, "Partido": f"{fx['local']} vs {fx['visita']}", "Fecha": fx.get("fecha", ""), **row})
            except Exception as exc:
                errores.append(f"{liga} · {fx.get('local','?')} vs {fx.get('visita','?')}: {type(exc).__name__}: {exc}")
    return _clean({"rows": rows, "errors": errores[:50]})


HTML = r'''<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FUT Europa</title><style>
:root{font-family:Inter,system-ui,Arial,sans-serif;color:#e8eef8;background:#07101f}body{margin:0;background:linear-gradient(180deg,#07101f,#0b1830);min-height:100vh}.wrap{max-width:1180px;margin:auto;padding:24px}.hero{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:20px}.hero h1{margin:0;font-size:32px}.muted{color:#9fb0c8}.badge{background:#143257;border:1px solid #2b5b91;padding:7px 10px;border-radius:999px}.panel{background:#0d1b2e;border:1px solid #223a58;border-radius:16px;padding:18px;margin:14px 0;box-shadow:0 10px 30px #0004}.row{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{background:#0a1526;border:1px solid #203855;border-radius:12px;padding:14px}.value{font-size:26px;font-weight:700;margin-top:5px}select,button{border-radius:10px;border:1px solid #365577;background:#10243d;color:#fff;padding:12px;font-size:15px}button{cursor:pointer;background:#1769aa;font-weight:700}button:hover{filter:brightness(1.12)}button:disabled{opacity:.55;cursor:wait}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px;border-bottom:1px solid #20344d;text-align:left}th{color:#9fc7ec}.error{color:#ff9c9c}.ok{color:#8fe0b0}.loader{display:none}.loader.on{display:inline}.meta{font-size:13px;color:#9fb0c8;margin:12px 0}.tabs{display:flex;gap:8px;flex-wrap:wrap}.tab{background:#10243d}.tab.active{background:#1769aa}@media(max-width:760px){.row,.cards{grid-template-columns:1fr}.hero{align-items:flex-start;flex-direction:column}}
</style></head><body><div class="wrap">
<div class="hero"><div><h1>⚽ FUT Europa</h1><div class="muted">FastAPI · Cloud Run · Elo · ML calibrado · Monte Carlo · EV+</div></div><div class="badge">Runtime web nativo</div></div>
<div class="panel"><div class="tabs" id="tabs"></div></div>
<div class="panel"><div class="row"><select id="fixture"></select><button id="analyze">Ejecutar análisis</button><button id="scan">Escáner Global EV+</button></div><div class="meta" id="msg">Selecciona una liga y un partido.</div></div>
<div id="result"></div><div id="scanResult"></div>
</div><script>
const leagues=['Premier League','La Liga','Serie A','Bundesliga','Ligue 1'];let league=leagues[0];
const tabs=document.getElementById('tabs'),fixture=document.getElementById('fixture'),msg=document.getElementById('msg'),result=document.getElementById('result');
function esc(s){return String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]))}
function pct(v){return v==null?'—':esc(v)+'%'}
function drawTabs(){tabs.innerHTML=leagues.map(x=>`<button class="tab ${x===league?'active':''}" data-l="${x}">${x}</button>`).join('');tabs.querySelectorAll('button').forEach(b=>b.onclick=()=>{league=b.dataset.l;drawTabs();loadFixtures()})}
async function loadFixtures(){fixture.innerHTML='<option>Cargando...</option>';msg.textContent='Cargando fixtures...';result.innerHTML='';try{const r=await fetch('/api/fixtures/'+encodeURIComponent(league));const d=await r.json();fixture.innerHTML=d.length?d.map(x=>`<option value="${esc(x.key)}">${esc(x.fecha)} | ${esc(x.local)} vs ${esc(x.visita)}</option>`).join(''):'<option value="">Sin fixtures</option>';msg.textContent=d.length?`${d.length} partidos disponibles.`:'No hay fixtures próximos disponibles.'}catch(e){msg.innerHTML='<span class="error">Error cargando fixtures.</span>'}}
function table(rows){if(!rows?.length)return '<div class="muted">No hay oportunidades EV+ con cuota real.</div>';const cols=Object.keys(rows[0]);return `<div style="overflow:auto"><table><thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${esc(r[c])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}
document.getElementById('analyze').onclick=async function(){if(!fixture.value)return;this.disabled=true;msg.textContent='Ejecutando motores...';const t=performance.now();try{const r=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({liga:league,fixture_key:fixture.value})});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Error');const mc=d.mc?.Resultado_1X2||{},ml=d.ml?.Resultado_1X2||{};result.innerHTML=`<div class="panel"><div class="cards"><div class="card">Local MC<div class="value">${pct(mc['Gana Local'])}</div></div><div class="card">Empate MC<div class="value">${pct(mc['Empate'])}</div></div><div class="card">Visita MC<div class="value">${pct(mc['Gana Visita'])}</div></div></div><div class="cards" style="margin-top:12px"><div class="card">Local ML<div class="value">${pct(ml['Gana Local'])}</div></div><div class="card">Empate ML<div class="value">${pct(ml['Empate'])}</div></div><div class="card">Visita ML<div class="value">${pct(ml['Gana Visita'])}</div></div></div><div class="meta">Mapeo: ${esc(d.meta.local_modelo)} vs ${esc(d.meta.visita_modelo)} · Elo ${esc(d.meta.elo_local)} / ${esc(d.meta.elo_visita)} · ML n=${esc(d.meta.train_rows)}</div><h3>Mercados y valor</h3>${table(d.bets)}</div>`;msg.innerHTML=`<span class="ok">Análisis terminado en ${((performance.now()-t)/1000).toFixed(1)} s.</span>`}catch(e){msg.innerHTML=`<span class="error">${esc(e.message)}</span>`}finally{this.disabled=false}}
document.getElementById('scan').onclick=async function(){this.disabled=true;document.getElementById('scanResult').innerHTML='<div class="panel">Escaneando las 5 ligas...</div>';try{const r=await fetch('/api/scan');const d=await r.json();document.getElementById('scanResult').innerHTML=`<div class="panel"><h3>Escáner Global EV+</h3>${table(d.rows)}${d.errors?.length?`<details><summary>${d.errors.length} avisos</summary><div class="meta">${d.errors.map(esc).join('<br>')}</div></details>`:''}</div>`}catch(e){document.getElementById('scanResult').innerHTML=`<div class="panel error">${esc(e.message)}</div>`}finally{this.disabled=false}}
drawTabs();loadFixtures();
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML)
