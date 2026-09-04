import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

import web_app as core
from modules.fut_sheet_ledger import persist_recommendations, sheets_diagnostic
from modules.fut_sheet_settlement import settle_pending_sheet

app = FastAPI(title="FUT Europa", version="2.9-resilient-timeout")


def _analizar_partido_linea_real(nombre_liga, fixture):
    """Núcleo productivo: una sola fuente de verdad para líneas: bookmaker -> ML + MC."""
    df = core.cargar_historico_liga(nombre_liga)
    if df.empty:
        raise ValueError("Sin histórico disponible")
    loc_api, vis_api = fixture["local"], fixture["visita"]
    loc, vis = core.resolver_nombre(loc_api, df), core.resolver_nombre(vis_api, df)
    tabla, ml, ml_ok = core.obtener_motores(nombre_liga, df)
    e_loc, e_vis = core.rating(tabla, loc), core.rating(tabla, vis)
    odds = core.obtener_cuotas_europa(fixture.get("fixture_id"), nombre_liga, loc_api, vis_api)
    lineas_casino = odds.get("_lineas", {}) if isinstance(odds, dict) else {}

    mc = core.simular_partido_europa(loc, vis, df, e_loc, e_vis, lineas_casino=lineas_casino)
    preds = ml.predecir_mercados_completos(
        loc, vis, elo_local=e_loc, elo_visita=e_vis, cuotas_1x2=odds,
        fecha_partido=fixture.get("fecha"), lineas_casino=lineas_casino
    ) if ml_ok else {}

    bets = core.analizar_apuestas_europa(
        mc, preds, fixture.get("fixture_id"), cuotas_personalizadas=odds,
        nombre_liga=nombre_liga, local=loc_api, visita=vis_api
    )
    mm = preds.get("Meta", {}) if isinstance(preds, dict) else {}
    ml_lines = mm.get("lineas_modeladas", {}) if isinstance(mm, dict) else {}
    line_lock = {}
    for tipo, line in lineas_casino.items():
        try:
            line_lock[tipo] = abs(float(ml_lines.get(tipo)) - float(line)) < 1e-9
        except Exception:
            line_lock[tipo] = False

    meta = {
        "local_modelo": loc, "visita_modelo": vis,
        "elo_local": round(e_loc, 1), "elo_visita": round(e_vis, 1),
        "ml_ok": ml_ok, "train_rows": getattr(ml, "n_train", 0),
        "temperature_1x2": mm.get("temperature_1x2", 1.0),
        "market_model_weight": mm.get("market_model_weight", 1.0),
        "market_blend_used": mm.get("market_blend_used", False),
        "rest_local": mm.get("rest_local", 7.0), "rest_visita": mm.get("rest_visita", 7.0),
        "lineas_casino": lineas_casino, "lineas_ml": ml_lines, "line_lock": line_lock,
        "line_status": odds.get("_line_status", {}) if isinstance(odds, dict) else {},
        "bookmaker": odds.get("_bookmaker") if isinstance(odds, dict) else None,
        "pricing_mode": odds.get("_pricing_mode", "unavailable") if isinstance(odds, dict) else "unavailable",
        "persist_allowed": bool(odds.get("_persist_allowed")) if isinstance(odds, dict) else False,
        "mc_uses_casino_line": bool(lineas_casino),
    }
    bet_rows = [] if bets is None or bets.empty else bets.replace({np.nan: None}).to_dict(orient="records")
    return core._clean({"mc": mc, "ml": preds, "bets": bet_rows, "meta": meta})


core.analizar_partido = _analizar_partido_linea_real


@app.get("/health")
def health():
    return {
        "status": "ok", "runtime": "fastapi", "streamlit": False,
        "scanner": "sequential_timeout_retry", "scanner_timeout_seconds": 180,
        "scanner_retry_timeout_seconds": 240, "scanner_retry_failed_once": True,
        "sheets_ledger": True, "automatic_settlement": True,
        "casino_line_montecarlo": True,
        "bookmaker_policy": "playdoit_strict_bet365_reference",
        "ambiguous_totals_blocked": True, "line_lock": True, "version": "2.9",
    }


@app.get("/api/sheets-status")
def sheets_status():
    return sheets_diagnostic()


@app.post("/api/settle")
def settle():
    result = settle_pending_sheet()
    if not result.get("ok"):
        raise HTTPException(status_code=503, detail=result)
    return result


@app.get("/api/status")
def status():
    return {liga: {"partidos": len(core.cargar_historico_liga(liga)), "ready": len(core.cargar_historico_liga(liga)) >= 150} for liga in core.LIGAS_IDS}


@app.get("/api/fixtures/{liga}")
def fixtures(liga: str):
    if liga not in core.LIGAS_IDS:
        raise HTTPException(404, "Liga no válida")
    data = core.obtener_proximos_partidos_europa(core.LIGAS_IDS[liga])
    return [{"key": k, **v} for k, v in data.items()]


def _resolve_fixture(req: core.AnalyzeRequest):
    if req.liga not in core.LIGAS_IDS:
        raise HTTPException(404, "Liga no válida")
    fixtures_map = core.obtener_proximos_partidos_europa(core.LIGAS_IDS[req.liga])
    fixture = fixtures_map.get(req.fixture_key)
    if not fixture:
        raise HTTPException(404, "Fixture no encontrado")
    return fixture


@app.post("/api/analyze")
def analyze(req: core.AnalyzeRequest):
    fixture = _resolve_fixture(req)
    try:
        return core.analizar_partido(req.liga, fixture)
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/scan-one")
def scan_one(req: core.AnalyzeRequest):
    fixture = _resolve_fixture(req)
    try:
        result = core.analizar_partido(req.liga, fixture)
        meta = result.get("meta", {}) if isinstance(result, dict) else {}
        persist_allowed = bool(meta.get("persist_allowed"))
        good = [r for r in result.get("bets", []) if "🔥" in str(r.get("Veredicto", "")) or "✅" in str(r.get("Veredicto", ""))]
        official_good = good if persist_allowed else []
        rows = [{"Liga": req.liga, "Partido": f"{fixture['local']} vs {fixture['visita']}", "Fecha": fixture.get("fecha", ""), **row} for row in official_good]
        ledger = {"ok": True, "written": 0, "skipped": 0}
        if official_good:
            ledger = persist_recommendations(req.liga, fixture, official_good)
        return core._clean({
            "ok": True, "liga": req.liga, "partido": f"{fixture['local']} vs {fixture['visita']}",
            "rows": rows, "written": int(ledger.get("written", 0) or 0),
            "skipped": int(ledger.get("skipped", 0) or 0), "ledger_ok": bool(ledger.get("ok", False)),
            "ledger_error": ledger.get("error"), "bookmaker": meta.get("bookmaker"),
            "pricing_mode": meta.get("pricing_mode"), "persist_allowed": persist_allowed,
            "blocked_reference_picks": len(good) - len(official_good),
            "line_status": meta.get("line_status", {}), "line_lock": meta.get("line_lock", {}),
        })
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc


def _json_line(payload):
    return json.dumps(core._clean(payload), ensure_ascii=False) + "\n"


@app.get("/api/scan-stream")
def scan_stream():
    def generate():
        total = 0
        league_fixtures = {}
        for liga, league_id in core.LIGAS_IDS.items():
            fixtures_map = core.obtener_proximos_partidos_europa(league_id)
            league_fixtures[liga] = list(fixtures_map.values())
            total += len(fixtures_map)
        yield _json_line({"type": "start", "total": total})
        done = 0
        for liga, fixtures_list in league_fixtures.items():
            for fx in fixtures_list:
                try:
                    core.analizar_partido(liga, fx)
                except Exception:
                    pass
                done += 1
                yield _json_line({"type": "progress", "liga": liga, "done": done, "total": total})
        yield _json_line({"type": "done", "done": done, "total": total})
    return StreamingResponse(generate(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


SCAN_OVERRIDE = r"""
async function scanFetch(item, timeoutMs){
  const controller=new AbortController();
  const timer=setTimeout(()=>controller.abort(),timeoutMs);
  try{
    const r=await fetch('/api/scan-one',{
      method:'POST',headers:{'Content-Type':'application/json'},cache:'no-store',
      body:JSON.stringify({liga:item.liga,fixture_key:item.key}),signal:controller.signal
    });
    let d=null;try{d=await r.json()}catch(_e){}
    if(!r.ok)throw new Error((d&&d.detail)?String(d.detail):('HTTP '+r.status));
    return d;
  }catch(e){
    if(e&&e.name==='AbortError')throw new Error(`timeout después de ${Math.round(timeoutMs/1000)} s`);
    throw e;
  }finally{clearTimeout(timer);}
}

document.getElementById('scan').onclick=async function(){
 const btn=this,box=document.getElementById('scanResult'),leagues=['Premier League','La Liga','Serie A','Bundesliga','Ligue 1'];
 btn.disabled=true;btn.textContent='Escaneando...';
 box.innerHTML='<div class="panel"><h3>Escáner Global EV+</h3><div id="scanProgress">Cargando partidos de la semana...</div><div class="meta" id="scanMeta"></div></div>';
 try{
  const queue=[],fixtureErrors=[];
  for(const liga of leagues){
   try{const r=await fetch('/api/fixtures/'+encodeURIComponent(liga),{cache:'no-store'});if(!r.ok)throw new Error('HTTP '+r.status);const items=await r.json();for(const fx of items)queue.push({liga:liga,key:fx.key,partido:`${fx.local} vs ${fx.visita}`});}
   catch(e){fixtureErrors.push(`${liga}: ${e.message}`)}
  }
  const total=queue.length;if(!total)throw new Error('No se encontraron partidos para analizar.');
  let done=0,saved=0,skipped=0,blocked=0;
  const rows=[],errors=[...fixtureErrors],failed=[];
  const p=document.getElementById('scanProgress'),m=document.getElementById('scanMeta');
  p.textContent=`0 / ${total} partidos analizados`;

  const absorb=(d,item)=>{
    if(Array.isArray(d.rows))rows.push(...d.rows);
    saved+=Number(d.written||0);skipped+=Number(d.skipped||0);blocked+=Number(d.blocked_reference_picks||0);
    if(d.ledger_ok===false)errors.push(`Sheets · ${item.liga} · ${item.partido}: ${d.ledger_error||'error'}`);
  };

  for(const item of queue){
   const pct=Math.round(done*100/total);
   p.innerHTML=`<b>${done} / ${total}</b> (${pct}%) · Analizando ${esc(item.liga)} · ${esc(item.partido)}`;
   m.textContent=`Oficiales Playdoit: ${rows.length} · Guardadas: ${saved} · Referencias bloqueadas: ${blocked} · Fallidos pendientes: ${failed.length} · Avisos: ${errors.length}`;
   try{const d=await scanFetch(item,180000);absorb(d,item);}
   catch(e){failed.push(item);errors.push(`${item.liga} · ${item.partido}: ${e.message}`);}
   done++;
   const pct2=Math.round(done*100/total);
   p.innerHTML=`<b>${done} / ${total}</b> (${pct2}%) · Último: ${esc(item.liga)} · ${esc(item.partido)}`;
   m.textContent=`Oficiales Playdoit: ${rows.length} · Guardadas: ${saved} · Referencias bloqueadas: ${blocked} · Fallidos pendientes: ${failed.length} · Avisos: ${errors.length}`;
  }

  const retryFailed=[];
  if(failed.length){
    for(let i=0;i<failed.length;i++){
      const item=failed[i];
      p.innerHTML=`Reintento ${i+1}/${failed.length} · ${esc(item.liga)} · ${esc(item.partido)}`;
      m.textContent=`Reintentando solo fallidos · Oficiales: ${rows.length} · Guardadas: ${saved}`;
      try{const d=await scanFetch(item,240000);absorb(d,item);}
      catch(e){retryFailed.push(item);errors.push(`REINTENTO · ${item.liga} · ${item.partido}: ${e.message}`);}
    }
  }

  let settlement={ok:false,settled:0},sh={};
  try{const sr=await fetch('/api/settle',{method:'POST',cache:'no-store'});settlement=await sr.json();if(!sr.ok)errors.push('Settlement: '+JSON.stringify(settlement.detail||settlement));}catch(e){errors.push('Settlement: '+e.message)}
  try{const hr=await fetch('/api/sheets-status',{cache:'no-store'});sh=await hr.json();}catch(e){errors.push('Sheets status: '+e.message)}
  const sheetLine=`Sheets: ${sh.ok?'conectado':'ERROR'} · filas existentes: ${sh.existing_rows??'—'} · schema: ${sh.schema_ok?'OK':'revisar'} · liquidadas: ${settlement.settled||0}`;
  box.innerHTML=`<div class="panel"><h3>Escáner Global EV+</h3><div class="ok">Escaneo terminado: ${done}/${total} partidos · ${rows.length} picks oficiales Playdoit · ${saved} nuevos guardados · ${blocked} referencias Bet365 bloqueadas · ${retryFailed.length} fallidos definitivos.</div><div class="meta">${esc(sheetLine)}</div><div style="margin-top:12px">${table(rows)}</div>${errors.length?`<details open><summary>${errors.length} avisos</summary><div class="meta error">${errors.map(esc).join('<br>')}</div></details>`:''}</div>`;
 }catch(e){box.innerHTML=`<div class="panel error">Error del escáner: ${esc(e.message)}</div>`;}
 finally{btn.disabled=false;btn.textContent='Escáner Global EV+';}
}
"""

HTML = core.HTML.replace("</body>", f"<script>{SCAN_OVERRIDE}</script></body>")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML, headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"})
