import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

import web_app as core
from modules.fut_sheet_ledger import persist_recommendations, sheets_diagnostic

app = FastAPI(title="FUT Europa", version="2.3-sheets-diagnostic")


@app.get("/health")
def health():
    return {"status": "ok", "runtime": "fastapi", "streamlit": False, "scanner": "streaming", "sheets_ledger": True, "version": "2.3"}


@app.get("/api/sheets-status")
def sheets_status():
    return sheets_diagnostic()


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

        yield _json_line({"type": "done", "rows": rows, "errors": errors[:50], "done": done, "total": total, "saved": saved, "skipped": skipped, "sheets": sheets_diagnostic()})

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
    const sheetLine=`Sheets: ${sh.ok?'conectado':'ERROR'} · filas existentes: ${sh.existing_rows??'—'} · schema: ${sh.schema_ok?'OK':'revisar'}`;
    box.innerHTML=`<div class="panel"><h3>Escáner Global EV+</h3><div class="ok">Escaneo terminado: ${finalData.done}/${finalData.total} partidos · ${finalData.saved||0} picks nuevos guardados en Sheets · ${finalData.skipped||0} duplicados omitidos.</div><div class="meta">${esc(sheetLine)}</div><div style="margin-top:12px">${table(finalData.rows)}</div>${finalData.errors?.length?`<details open><summary>${finalData.errors.length} avisos</summary><div class="meta error">${finalData.errors.map(esc).join('<br>')}</div></details>`:''}</div>`;
  }catch(e){box.innerHTML=`<div class="panel error">Error del escáner: ${esc(e.message)}</div>`}
  finally{btn.disabled=false;btn.textContent='Escáner Global EV+'}
}
"""

# Always append the override after core's own script; this avoids fragile string matching.
HTML = core.HTML.replace("</body>", f"<script>{SCAN_OVERRIDE}</script></body>")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML, headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"})
