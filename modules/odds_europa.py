import os
import re
import requests
import pandas as pd
import unicodedata

API_KEY=os.environ.get("API_SPORTS_KEY"); BASE_URL="https://v3.football.api-sports.io"; HEADERS={"x-apisports-key":API_KEY} if API_KEY else {}
PRIMARY_BOOKMAKER="bet365"; CALIBRATION_VERSION="strict-v4-bet365-line-lock"
SUPPORTED_HALF_LINES={"goles":{1.5,2.5,3.5,4.5},"corners":{7.5,8.5,9.5,10.5,11.5,12.5},"tarjetas":{2.5,3.5,4.5,5.5,6.5,7.5}}
MAX_DISAGREEMENT_PP=8.0; MIN_MODEL_PROB_PCT=60.0; STRONG_EV_PCT=10.0; MODERATE_EV_PCT=5.0; STRONG_KELLY_PCT=1.0; MODERATE_KELLY_PCT=0.75; KELLY_FRACTION=0.10


def american_to_decimal(american):
    try: american=float(american)
    except Exception:return 0.0
    if american==0:return 0.0
    return round((american/100.0)+1.0,3) if american>0 else round((100.0/abs(american))+1.0,3)


def normalizar_nombre(nombre): return unicodedata.normalize("NFKD",str(nombre)).encode("ASCII","ignore").decode("utf-8").lower().strip()

def _extract_line(value):
    m=re.search(r"(?:Over|Under)\s*([0-9]+(?:\.[0-9]+)?)",str(value),flags=re.I); return float(m.group(1)) if m else None


def _select_bookmaker(bookmakers):
    normalized=[(normalizar_nombre(b.get("name","")),b) for b in (bookmakers or [])]
    primary=next((b for name,b in normalized if PRIMARY_BOOKMAKER in name),None)
    if primary:return primary,"bet365",True
    return None,"unavailable",False


def _supported_line(tipo,line):
    try:return round(float(line),2) in SUPPORTED_HALF_LINES.get(tipo,set())
    except Exception:return False


def _balanced_supported_line(tipo, price_map):
    """Pick the most market-central supported line from complete O/U pairs.

    Pre-match odds can contain several valid total lines. API-Football does not expose
    the live `main` flag on pre-match values, so use price balance only among lines
    that are directly supported by the trained model. Lower score = more balanced.
    Fail closed on exact score ties so we never guess between equally plausible lines.
    """
    candidates=[]
    for line,sides in (price_map or {}).items():
        try:
            line=float(line); over=float(sides.get("Over",0)); under=float(sides.get("Under",0))
        except Exception:
            continue
        if over<=1.01 or under<=1.01 or not _supported_line(tipo,line):
            continue
        io,iu=1.0/over,1.0/under
        total=io+iu
        if total<=0:continue
        p_over=io/total
        balance=abs(p_over-0.5)
        symmetry=abs(over-under)
        candidates.append((round(balance,8),round(symmetry,8),line))
    if not candidates:return None,"missing_supported"
    candidates.sort(key=lambda x:(x[0],x[1],x[2]))
    best=candidates[0]
    tied=[c for c in candidates if abs(c[0]-best[0])<1e-8 and abs(c[1]-best[1])<1e-8]
    if len(tied)>1:return None,f"balanced_tie:{len(tied)}"
    return best[2],f"balanced_supported:{len(candidates)}"


def obtener_cuotas_europa(fixture_id,nombre_liga=None,local=None,visita=None):
    cuotas={"1":0.0,"X":0.0,"2":0.0,"_lineas":{},"_bookmaker":None,"_pricing_mode":"unavailable","_persist_allowed":False,"_line_status":{},"_calibration":CALIBRATION_VERSION}
    if not API_KEY or fixture_id in (None,999999):return cuotas
    try:
        response=requests.get(f"{BASE_URL}/odds",headers=HEADERS,params={"fixture":str(fixture_id)},timeout=8)
        if response.status_code!=200:return cuotas
        data=response.json().get("response",[])
        if not data:return cuotas
        bookmakers=data[0].get("bookmakers",[]); bm,pricing_mode,persist_allowed=_select_bookmaker(bookmakers)
        if not bm:return cuotas
        cuotas["_bookmaker"]=bm.get("name"); cuotas["_pricing_mode"]=pricing_mode; cuotas["_persist_allowed"]=persist_allowed
        line_prices={"goles":{},"corners":{},"tarjetas":{}}
        for mercado in bm.get("bets",[]):
            mid=mercado.get("id"); bet_name=normalizar_nombre(mercado.get("name",""))
            if mid==5 or ("goal" in bet_name and ("over" in bet_name or "under" in bet_name)):tipo="goles"
            elif mid==45 or "corner" in bet_name:tipo="corners"
            elif "card" in bet_name or "booking" in bet_name or "tarjeta" in bet_name:tipo="tarjetas"
            else:tipo=None
            for valor in mercado.get("values",[]):
                key=str(valor.get("value","")).strip()
                try:odd=float(valor.get("odd"))
                except Exception:continue
                if mid==1 and key=="Home":cuotas["1"]=odd
                elif mid==1 and key=="Draw":cuotas["X"]=odd
                elif mid==1 and key=="Away":cuotas["2"]=odd
                elif tipo and key.lower().startswith(("over","under")):
                    line=_extract_line(key)
                    if line is None:continue
                    suffix="Goles" if tipo=="goles" else "Corners" if tipo=="corners" else "Tarjetas"; side="Over" if key.lower().startswith("over") else "Under"
                    cuotas[f"{side} {line:g} {suffix}"]=odd; line_prices[tipo].setdefault(float(line),{})[side]=odd
        for tipo,price_map in line_prices.items():
            chosen,status=_balanced_supported_line(tipo,price_map)
            if chosen is not None:
                cuotas["_lineas"][tipo]=chosen
                cuotas["_line_status"][tipo]=status
            else:
                complete=[]
                for line,sides in price_map.items():
                    try:
                        if float(sides.get("Over",0))>1.01 and float(sides.get("Under",0))>1.01:complete.append(float(line))
                    except Exception:pass
                if status.startswith("balanced_tie"):
                    cuotas["_line_status"][tipo]=status
                elif complete:
                    cuotas["_line_status"][tipo]="unsupported_or_no_supported_complete"
                else:
                    cuotas["_line_status"][tipo]="missing"
    except Exception:return cuotas
    return cuotas


def calcular_kelly_fraccional(prob_modelo_decimal,cuota_decimal,fraccion=KELLY_FRACTION):
    if cuota_decimal<=1.0 or not 0<prob_modelo_decimal<1:return 0.0
    b=cuota_decimal-1.0; kelly=((b*prob_modelo_decimal)-(1.0-prob_modelo_decimal))/b
    return round(max(0.0,kelly)*fraccion*100.0,2)


def _mc_market_probability(resultados_mc,tipo,side,line):
    label="Goles" if tipo=="goles" else "Corners" if tipo=="corners" else "Tarjetas"
    try:return float(resultados_mc.get("Lineas_Casino",{}).get(tipo,{}).get(f"{side} {line:g} {label}",0.0))
    except Exception:return 0.0


def _line_locked(cuotas,preds_ml,resultados_mc,tipo,line,side):
    """Fail closed: sportsbook, ML metadata and MC exact market must all agree."""
    try:
        sportsbook=float(cuotas.get("_lineas",{}).get(tipo)); ml=float(preds_ml.get("Meta",{}).get("lineas_modeladas",{}).get(tipo))
        if abs(sportsbook-line)>1e-9 or abs(ml-line)>1e-9:return False
        label="Goles" if tipo=="goles" else "Corners" if tipo=="corners" else "Tarjetas"
        key=f"{side} {line:g} {label}"
        return key in resultados_mc.get("Lineas_Casino",{}).get(tipo,{})
    except Exception:return False


def analizar_apuestas_europa(resultados_mc,preds_ml,fixture_id,cuotas_personalizadas=None,nombre_liga=None,local=None,visita=None):
    cuotas=cuotas_personalizadas if cuotas_personalizadas is not None else obtener_cuotas_europa(fixture_id,nombre_liga,local,visita)
    if not cuotas:return pd.DataFrame()
    def get_prob(d,cat,key):
        try:return float(d.get(cat,{}).get(key,0.0))
        except Exception:return 0.0
    mercados=[{"nombre":"Gana Local","cat":"Resultado_1X2","key":"Gana Local","odd_key":"1","tipo":"1x2"},{"nombre":"Empate","cat":"Resultado_1X2","key":"Empate","odd_key":"X","tipo":"1x2"},{"nombre":"Gana Visita","cat":"Resultado_1X2","key":"Gana Visita","odd_key":"2","tipo":"1x2"}]
    lineas=cuotas.get("_lineas",{}) if isinstance(cuotas.get("_lineas",{}),dict) else {}
    for tipo in ("goles","corners","tarjetas"):
        try:line=float(lineas[tipo])
        except Exception:continue
        label="Goles" if tipo=="goles" else "Corners" if tipo=="corners" else "Tarjetas"
        for side in ("Over","Under"):
            odd_key=f"{side} {line:g} {label}"; mercados.append({"nombre":odd_key,"odd_key":odd_key,"tipo":tipo,"line":line,"side":side})
    rows=[]; official=bool(cuotas.get("_persist_allowed")); bookmaker=cuotas.get("_bookmaker") or "sin bookmaker"; pricing_mode=cuotas.get("_pricing_mode","unavailable")
    for m in mercados:
        try:cuota=float(cuotas.get(m["odd_key"],0.0))
        except Exception:cuota=0.0
        if cuota<=1.01:continue
        if m["tipo"]=="1x2":prob_mc=get_prob(resultados_mc,m["cat"],m["key"]); prob_ml=get_prob(preds_ml,m["cat"],m["key"])
        else:
            line=float(m["line"])
            if not _line_locked(cuotas,preds_ml,resultados_mc,m["tipo"],line,m["side"]):continue
            prob_mc=_mc_market_probability(resultados_mc,m["tipo"],m["side"],line)
            if m["tipo"]=="goles":cat,key="Goles_Over_Under",f"{m['side']} {line:g}"
            elif m["tipo"]=="corners":cat,key="Corners_Totales",f"{m['side']} {line:g} Corners"
            else:cat,key="Tarjetas_Totales",f"{m['side']} {line:g} Tarjetas"
            prob_ml=get_prob(preds_ml,cat,key)
        if prob_mc<=0 or prob_ml<=0:continue
        disagreement=abs(prob_mc-prob_ml); prob_modelo_pct=0.65*prob_ml+0.35*prob_mc; p=prob_modelo_pct/100.0; ev_pct=((p*cuota)-1.0)*100.0; kelly=calcular_kelly_fraccional(p,cuota)
        if disagreement>MAX_DISAGREEMENT_PP:verdict="❌ NO BET — modelos en desacuerdo";kelly=0.0
        elif min(prob_mc,prob_ml)<MIN_MODEL_PROB_PCT:verdict="❌ NO BET — confianza insuficiente";kelly=0.0
        elif ev_pct>=STRONG_EV_PCT and kelly>=STRONG_KELLY_PCT:verdict="🔥 Value Fuerte"
        elif ev_pct>=MODERATE_EV_PCT and kelly>=MODERATE_KELLY_PCT:verdict="✅ Value Moderado"
        elif ev_pct>0:verdict="⚠️ EV Positivo Marginal";kelly=0.0
        else:verdict="❌ EV Negativo";kelly=0.0
        if not official and ("🔥" in verdict or "✅" in verdict):verdict="🧪 BOOKMAKER NO OFICIAL";kelly=0.0
        rows.append({"Mercado":m["nombre"],"Bookmaker":bookmaker,"Modo cuota":pricing_mode,"Calibración":CALIBRATION_VERSION,"Fuente prob.":"ML line-aware + MC line-locked","Prob. MC":f"{prob_mc:.1f}%","Prob. ML":f"{prob_ml:.1f}%","Prob. usada":f"{prob_modelo_pct:.1f}%","Desacuerdo pp":round(disagreement,1),"Cuota real":round(cuota,3),"EV (Valor)":f"{ev_pct:.2f}%","Stake Recomendado":f"{kelly:.2f}% Bank","Veredicto":verdict})
    return pd.DataFrame(rows)
