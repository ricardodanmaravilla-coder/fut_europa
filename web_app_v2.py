import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

import web_app as core
from modules.fut_sheet_ledger import persist_recommendations, sheets_diagnostic
from modules.fut_sheet_settlement import settle_pending_sheet

app = FastAPI(title="FUT Europa", version="2.5-casino-lines")


def _analizar_partido_linea_real(nombre_liga, fixture):
    """Núcleo productivo: Monte Carlo usa exactamente la línea O/U del bookmaker."""
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
        fecha_partido=fixture.get("fecha")
    ) if ml_ok else {}
    bets = core.analizar_apuestas_europa(
        mc, preds, fixture.get("fixture_id"), cuotas_personalizadas=odds,
        nombre_liga=nombre_liga, local=loc_api, visita=vis_api
    )
    mm = preds.get("Meta", {}) if isinstance(preds, dict) else {}
    meta = {
        "local_modelo": loc,
        "visita_modelo": vis,
        "elo_local": round(e_loc, 1),
        "elo_visita": round(e_vis, 1),
        "ml_ok": ml_ok,
        "train_rows": getattr(ml, "n_train", 0),
        "temperature_1x2": mm.get("temperature_1x2", 1.0),
        "market_model_weight": mm.get("market_model_weight", 1.0),
        "market_blend_used": mm.get("market_blend_used", False),
        "rest_local": mm.get("rest_local", 7.0),
        "rest_visita": mm.get("rest_visita", 7.0),
        "lineas_casino": lineas_casino,
        "bookmaker": odds.get("_bookmaker") if isinstance(odds, dict) else None,
        "mc_uses_casino_line": bool(lineas_casino),
    }
    bet_rows = [] if bets is None or bets.empty else bets.replace({np.nan: None}).to_dict(orient="records")
    return core._clean({"mc": mc, "ml": preds, "bets": bet_rows, "meta": meta})


# Todo el runtime V2 (análisis individual y escáner) pasa por el mismo núcleo corregido.
core.analizar_partido = _analizar_partido_linea_real


@app.get("/health")
def health():
    return {
        "status": "ok",
        "runtime": "fastapi",
        "streamlit": False,
        "scanner": "streaming",
        "sheets_ledger": True,
        "automatic_settlement": True,
        "casino_line_montecarlo": True,
        "version": "2.5",
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


@app.post("/api/analyze")
def analyze(req: core.AnalyzeRequest):
    if req.liga not in core.LIGAS_IDS:
        raise HTTPException(404, "Liga no válida")
    fixtures_map = core.obtener_proximos_partidos_europa(core.LIGAS_IDS[req.liga])
    fixture = fixtures_map.get(req.fixture_key)
    if not fixture:
        raise HTTPException(404, "Fixture no encontrado")
    try:
        return core.analizar_partido(req.liga, fixture)
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

        diag = sheets_diagnostic()
        yield _json_line({"type": "start", "total": total, "leagues": len(core.LIGAS_IDS), "sheets": diag})

        rows = []
        errors = []
        saved = 0
        skipped = 0
        done = 0
        if not diag.get("ok"):
            errors.append(f"Sheets diagnóstico: {diag.get('error', 'sin acceso')}")

        for liga, fixtures_list in league_fixtures.items():
            yield _json_line({"type": "league", "liga": liga, "count": len(fixtures_list), "done": done, "total": total})
            for fx in fixtures_list:
                try:
                    result = core.analizar_partido(liga, fx)
                    good = [r for r in result["bets"] if "🔥" in str(r.get("Veredicto", "")) or "✅" in str(r.get("Veredicto", ""))]
                    for row in good:
                        rows.append({"Liga": liga, "Partido": f"{fx['local']} vs {fx['visita']}", "Fecha": fx.get("fecha", ""), **row})
                    if good:
                        ledger = persist_recommendations(liga, fx, good)
                        saved += int(ledger.get("written", 0) or 0)
                        skipped += int(ledger.get("skipped", 0) or 0)
                        if not ledger.get("ok", False):
                            errors.append(f"Sheets · {liga} · {fx.get('local','?')} vs {fx.get('visita','?')}: {ledger.get('error','error desconocido')}")
                except Exception as exc:
                    errors.append(f"{liga} · {fx.get('local','?')} vs {fx.get('visita','?')}: {type(exc).__name__}: {exc}")
                done += 1
                yield _json_line({"type": "progress", "liga": liga, "partido": f"{fx.get('local','?')} vs {fx.get('visita','?')}", "done": done, "total": total, "found": len(rows), "saved": saved, "skipped": skipped, "errors": len(errors)})

        settlement = settle_pending_sheet()
        if not settlement.get("ok"):
            errors.append("Settlement: " + "; ".join(settlement.get("errors", [])[:3]))
        yield _json_line({
            "type": "done",
            "rows": rows,
            "errors": errors[:50],
            "done": done,
            "total": total,
            "saved": saved,
            "skipped": skipped,
            "sheets": sheets_diagnostic(),
            "settlement": settlement,
        })

    return StreamingResponse(generate(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


SCAN_OVERRIDE = r"""
document.getElementById('scan').onclick=async function(){
  const btn=this,box=document.getElementById('scanResult');
  btn.disabled=true;
  btn.textContent='Escaneando...';
  box.innerHTML='<div class="panel"><h3>Escáner Global EV+</h3><div id="scanProgress">Preparando fixtures...</div><div class="meta" id="scanMeta"></div></div>';
  try{
    const r=await fetch('/api/scan-stream',{cache:'no-store'});
    if(!r.ok)throw new Error('HTTP '+r.status);
    if(!r.body)throw new Error('El navegador no recibió el flujo del escáner.');
    const reader=r.body.getReader(),decoder=new TextDecoder();
    let buffer='',finalData=null;
    while(true){
      const packet=await reader.read();
      if(packet.done)break;
      buffer+=decoder.decode(packet.value,{stream:true});
      const lines=buffer.split('\n'); buffer=lines.pop();
      for(const line of lines){
        if(!line.trim())continue;
        const d=JSON.parse(line);
        const p=document.getElementById('scanProgress'),m=document.getElementById('scanMeta');
        if(d.type==='start'){
          p.textContent=`0 / ${d.total} partidos analizados`;
          const sh=d.sheets||{};
          m.textContent=`Sheets: ${sh.ok?'conectado':'ERROR'} · Cuenta: ${sh.service_account||'desconocida'}`;
        } else if(d.type==='league'){
          m.textContent=`${d.liga}: ${d.count} partidos encontrados.`;
        } else if(d.type==='progress'){
          const pct=d.total?Math.round(d.done*100/d.total):100;
          p.innerHTML=`<b>${d.done} / ${d.total}</b> (${pct}%) · ${esc(d.liga)} · ${esc(d.partido)}`;
          m.textContent=`Oportunidades: ${d.found} · Guardadas: ${d.saved||0} · Duplicadas: ${d.skipped||0} · Avisos: ${d.errors}`;
        } else if(d.type==='done') finalData=d;
      }
    }
    if(!finalData)throw new Error('El escáner terminó sin respuesta final.');
    const sh=finalData.sheets||{};
    const st=finalData.settlement||{};
    const sheetLine=`Sheets: ${sh.ok?'conectado':'ERROR'} · filas existentes: ${sh.existing_rows??'—'} · schema: ${sh.schema_ok?'OK':'revisar'} · liquidadas: ${st.settled||0}`;
    box.innerHTML=`<div class="panel"><h3>Escáner Global EV+</h3><div class="ok">Escaneo terminado: ${finalData.done}/${finalData.total} partidos · ${finalData.saved||0} picks nuevos guardados en Sheets · ${finalData.skipped||0} duplicados omitidos.</div><div class="meta">${esc(sheetLine)}</div><div style="margin-top:12px">${table(finalData.rows)}</div>${finalData.errors?.length?`<details open><summary>${finalData.errors.length} avisos</summary><div class="meta error">${finalData.errors.map(esc).join('<br>')}</div></details>`:''}</div>`;
  }catch(e){box.innerHTML=`<div class="panel error">Error del escáner: ${esc(e.message)}</div>`}
  finally{btn.disabled=false;btn.textContent='Escáner Global EV+'}
}
"""

HTML = core.HTML.replace("</body>", f"<script>{SCAN_OVERRIDE}</script></body>")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML, headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"})
