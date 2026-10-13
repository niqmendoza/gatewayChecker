#!/usr/bin/env python3
"""Pruebas de contexto y herramientas por Chat Completions. Python 3.10+.
Instalar: python -m pip install -r requirements.txt. Ver README.md.
No modifica el gateway. Las herramientas se simulan localmente.
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests
import tiktoken
from dotenv import load_dotenv


def load_config():
    # Nunca buscar .env en directorios padre; el entorno externo prevalece.
    load_dotenv(Path(__file__).resolve().with_name(".env"), override=False, interpolate=False)


def headers_for(key, auth):
    if auth not in ("bearer", "api-key"):
        raise ValueError("AUTH debe ser bearer o api-key")
    return {"Authorization": f"Bearer {key}"} if auth == "bearer" else {"api-key": key}


def endpoint_url(value, endpoint):
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password or parts.fragment:
        raise ValueError("URL debe ser http(s), sin credenciales incrustadas ni fragmento")
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[:-len("/chat/completions")]
    elif path.endswith("/models"):
        path = path[:-len("/models")]
    # LiteLLM admite /chat/completions y /v1/chat/completions.
    # Conservar el prefijo aportado por el usuario, incluido /openai/v1.
    return urlunsplit(parts._replace(path=path + "/" + endpoint))


def check_http(response):
    if 200 <= response.status_code < 300:
        return
    # No guardar URLs, claves ni cuerpos de error potencialmente sensibles.
    error = response.text.lower()
    code = response.status_code
    kind = {401: "auth", 403: "auth", 404: "endpoint", 413: "body_limit",
            429: "rate_limit"}.get(code, "server" if code >= 500 else "request")
    if code in (400, 422) and any(s in error for s in
            ("context_length_exceeded", "context window", "maximum context",
             "max_req_total_len", "max_req_input_len", "too many input tokens")):
        kind = "context_limit"
    raise RuntimeError(f"HTTP {code}: {kind}; revisar el error en los logs del servicio")


def discover_model(url, key, auth, timeout):
    with requests.get(endpoint_url(url, "models"), headers=headers_for(key, auth),
                      timeout=timeout, allow_redirects=False) as response:
        check_http(response)
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("/models debe devolver un objeto con data como lista; usar --gateway-model")
        data = payload["data"]
    ids = sorted({m["id"] for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)})
    matches = [m for m in ids if re.search(r"(?:^|[/_-])gpt[-_ ]?5[.]6[-_ ]sol(?:$|[/_-])", m, re.I)]
    if len(matches) == 1:
        return matches[0]
    # Listar identificadores, nunca las credenciales ni el cuerpo HTTP.
    raise ValueError("No hay un único GPT-5.6 Sol identificable en /models. "
                     "Elegí --gateway-model ID (o GATEWAY_MODEL). Modelos: " +
                     json.dumps(ids, ensure_ascii=False))


def configured_routes(args):
    routes = {}
    for name in ("gateway", "azure"):
        if args.route not in (name, "both"):
            continue
        prefix = name.upper()
        url, key = os.getenv(prefix + "_URL", "").strip(), os.getenv(prefix + "_KEY", "")
        if not url or not key:
            raise ValueError(f"Definir {prefix}_URL y {prefix}_KEY en .env o en el entorno")
        auth = os.getenv(prefix + "_AUTH", "bearer" if name == "gateway" else "api-key")
        headers_for(key, auth)
        endpoint = endpoint_url(url, "chat/completions")
        model = getattr(args, name + "_model") or os.getenv(prefix + "_MODEL", "").strip()
        if not model and name == "gateway":
            model = discover_model(url, key, auth, args.timeout)
        if not model:
            raise ValueError("Definir AZURE_MODEL con el nombre del despliegue, o --azure-model")
        routes[name] = (endpoint, model, key, auth)
    return routes


def json_object(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("La respuesta no es un objeto JSON")
    return value


def completion(route, body, timeout):
    url, model, key, auth = route
    headers = headers_for(key, auth)
    with requests.post(url, headers=headers, json=dict(body, model=model),
                       stream=body.get("stream", False), timeout=timeout,
                       allow_redirects=False) as response:
        check_http(response)
        if not body.get("stream"):
            data = response.json()
            choice = data["choices"][0]
            return choice["message"], {"model": data.get("model"),
                    "usage": data.get("usage"), "finish_reason": choice.get("finish_reason")}
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise ValueError("stream=true no devolvió text/event-stream")
        message, calls, meta = {"role": "assistant", "content": ""}, {}, {}
        done = False
        for raw in response.iter_lines():
            if not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                done = True
                break
            chunk = json.loads(payload)
            if "error" in chunk:
                raise ValueError("Error dentro del stream; revisar logs del gateway")
            if chunk.get("model"):
                meta["model"] = chunk["model"]
            if chunk.get("usage"):
                meta["usage"] = chunk["usage"]
            for choice in chunk.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                if choice.get("finish_reason"):
                    meta["finish_reason"] = choice["finish_reason"]
                delta = choice.get("delta", {})
                message["content"] += delta.get("content") or ""
                for part in delta.get("tool_calls", []):
                    call = calls.setdefault(part["index"], {"id": "", "type": "function",
                            "function": {"name": "", "arguments": ""}})
                    call["id"] += part.get("id") or ""
                    if part.get("type"):
                        call["type"] = part["type"]
                    for field in ("name", "arguments"):
                        call["function"][field] += part.get("function", {}).get(field) or ""
        if not done or not meta.get("finish_reason"):
            raise ValueError("Stream incompleto: falta DONE o finish_reason")
        if calls:
            message["tool_calls"] = [calls[i] for i in sorted(calls)]
        return message, meta


def context_prompt(size, seed, enc):
    rng = random.Random(seed)
    expected = {k: f"{rng.getrandbits(48):012x}" for k in ("start", "middle", "end")}
    instruction = ('Encuentra los registros TARGET_start, TARGET_middle y TARGET_end. '
                   'Devuelve solo JSON con start, middle, end y combined. '
                   'combined es start seguido de end, separados por dos puntos.\n')
    markers = [f"\nTARGET_{k}={v}\n" for k, v in expected.items()]
    budget = size - len(enc.encode(instruction + "".join(markers))) - 10
    # Datos distintos por ensayo; las rutas gateway y Azure reciben el mismo texto.
    filler = []
    while len(filler) < budget:
        rows = "".join(f"Registro {rng.randrange(10**8)}: valor={rng.randrange(10**8)}; estado=archivado.\n"
                       for _ in range(1000))
        filler.extend(enc.encode(rows))
    filler = filler[:budget]
    cuts = [0, int(budget * .05), int(budget * .50), int(budget * .95), budget]
    chunks = [enc.decode(filler[cuts[i]:cuts[i+1]]) for i in range(4)]
    text = instruction + chunks[0] + markers[0] + chunks[1] + markers[1] + chunks[2] + markers[2] + chunks[3]
    expected["combined"] = expected["start"] + ":" + expected["end"]
    return text, expected


TOOL = {"type": "function", "function": {"name": "add", "description": "Suma dos enteros.",
        "parameters": {"type": "object", "properties": {"a": {"type": "integer"},
        "b": {"type": "integer"}}, "required": ["a", "b"], "additionalProperties": False}}}


def tool_probe(route, choice, effort, stream, args, row, receipt):
    messages = [{"role": "user", "content": "Llama a add con a=19 y b=23. Después devuelve solo JSON "
                 "con sum y receipt, copiando ambos valores del resultado de la herramienta."}]
    body = {"messages": messages, "tools": [TOOL], "tool_choice": choice,
            "max_completion_tokens": args.output_tokens, "stream": stream}
    if effort != "default":
        body["reasoning_effort"] = effort
    if stream:
        body["stream_options"] = {"include_usage": True}
    first, meta = completion(route, body, args.timeout)
    row["first"] = meta
    row["stage"] = "tool_call"
    calls = first.get("tool_calls") or []
    if len(calls) != 1 or meta.get("finish_reason") != "tool_calls":
        raise ValueError("Se esperaba una tool_call estructurada con finish_reason=tool_calls")
    call = calls[0]
    if not isinstance(call.get("id"), str) or not call["id"] or call.get("type") != "function" or call["function"]["name"] != "add":
        raise ValueError("id, type o nombre de herramienta inválido")
    values = json_object(call["function"]["arguments"])
    if values != {"a": 19, "b": 23} or any(type(v) is not int for v in values.values()):
        raise ValueError("Argumentos de herramienta incorrectos")
    # No se ejecuta código del modelo: solamente esta suma conocida.
    result = {"sum": 42, "receipt": receipt}
    messages += [first, {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}]
    body["tool_choice"] = "none"
    row["stage"] = "tool_result"
    final, meta = completion(route, body, args.timeout)
    row["second"] = meta
    answer = json_object(final.get("content"))
    row["ok"] = all(answer.get(k) == v for k, v in result.items()) and meta.get("finish_reason") == "stop"
    if not row["ok"]:
        row["error"] = "No recuperó correctamente el resultado y recibo de la herramienta"


def main():
    load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "context", "tools"], default="all")
    parser.add_argument("--route", choices=["gateway", "azure", "both"], default="gateway")
    parser.add_argument("--gateway-model", help="ID exacto en LiteLLM; omite descubrimiento /models")
    parser.add_argument("--azure-model", help="Nombre del despliegue de Azure")
    parser.add_argument("--sizes", default="2048,8192,32768")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--encoding", help="Tokenizer explícito si el modelo no está registrado")
    parser.add_argument("--input-limit", type=int, default=922000)
    parser.add_argument("--output-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--out", default="reports/probe_results.json")
    parser.add_argument("--dry-run", action="store_true", help="Genera y cuenta contexto, sin llamadas API")
    args = parser.parse_args()
    try:
        sizes = sorted(set(int(s) for s in args.sizes.split(",")))
    except ValueError:
        parser.error("--sizes debe ser una lista de enteros separados por comas")
    if min(sizes) < 512 or args.repeats < 1 or not math.isfinite(args.timeout) or min(args.input_limit, args.output_tokens, args.timeout) <= 0:
        parser.error("Tamaños >=512; repeticiones, límites y timeout positivos")
    try:
        enc = tiktoken.get_encoding(args.encoding) if args.encoding else tiktoken.encoding_for_model("gpt-5.6-sol")
    except (KeyError, ValueError):
        parser.error("tiktoken no reconoce el modelo/encoding; usar --encoding o200k_base (conteo aproximado)")
    try:
        routes = {} if args.dry_run else configured_routes(args)
    except json.JSONDecodeError:
        parser.error("/models no devolvió JSON válido; verificar endpoint o usar --gateway-model")
    except requests.RequestException as error:
        parser.error("Descubrimiento /models: " + type(error).__name__ + "; verificar conexión o usar --gateway-model")
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if args.dry_run and args.mode == "tools":
        parser.error("--dry-run genera contexto; usar --mode context o all")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"encoding": enc.name, "explicit_encoding": bool(args.encoding),
              "reference_input_limit": args.input_limit, "dry_run": args.dry_run,
              "selected_models": {name: route[1] for name, route in routes.items()}, "tests": []}
    halted = set()

    def run(name, row, operation):
        row.update(route=name, ok=False)
        if name in halted:
            row.update(skipped=True, error="Ruta detenida por error previo")
            report["tests"].append(row)
            return
        start = time.monotonic()
        try:
            operation()
        except Exception as error:
            # Guardar clase y mensaje controlado, nunca respuestas crudas o claves.
            row["ok"] = False
            row["error"] = str(error) if isinstance(error, (RuntimeError, ValueError)) and not isinstance(error, (json.JSONDecodeError, requests.RequestException)) else type(error).__name__
            if isinstance(error, requests.RequestException) or any(s in row["error"] for s in
                    ("auth;", "endpoint;", "rate_limit;", "HTTP 5")):
                halted.add(name)
        row["seconds"] = round(time.monotonic() - start, 2)
        report["tests"].append(row)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if args.mode in ("all", "context"):
        for size in sizes:
            for trial in range(args.repeats):
                text, expected = context_prompt(size, args.seed + size + trial, enc)
                count = len(enc.encode(text))
                if args.dry_run:
                    row = {"test": "context", "target": size, "content_tokens": count, "trial": trial}
                    report["tests"].append(row)
                    print(json.dumps(row))
                    continue
                for name, route in routes.items():
                    row = {"test": "context", "target": size, "trial": trial, "content_tokens": count}
                    def context(route=route, row=row):
                        message, meta = completion(route, {"messages": [{"role": "user", "content": text}],
                                "max_completion_tokens": args.output_tokens, "reasoning_effort": "none"}, args.timeout)
                        row.update(meta)
                        answer = json_object(message.get("content"))
                        row["checks"] = {k: answer.get(k) == v for k, v in expected.items()}
                        row["ok"] = all(row["checks"].values()) and meta.get("finish_reason") == "stop"
                        reported = (meta.get("usage") or {}).get("prompt_tokens")
                        if reported is not None:
                            row["reported_minus_content"] = reported - count
                            row["reported_input_pct_of_reference_limit"] = round(100 * reported / args.input_limit, 2)
                    run(name, row, context)
    if args.mode in ("all", "tools") and not args.dry_run:
        for label, choice, effort, stream in [
                ("auto_none", "auto", "none", False),
                ("forced_none", {"type": "function", "function": {"name": "add"}}, "none", False),
                ("auto_default", "auto", "default", False),
                ("auto_none_stream", "auto", "none", True)]:
            receipt = os.urandom(8).hex()
            for name, route in routes.items():
                row = {"test": "tools", "case": label, "stage": "request"}
                run(name, row, lambda: tool_probe(route, choice, effort, stream, args, row, receipt))
    if args.dry_run:
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0
    report["context_summary"] = {}
    for name in routes:
        rows = [r for r in report["tests"] if r["route"] == name and r["test"] == "context"]
        accepted = [r for r in rows if r.get("finish_reason")]
        successful = [r for r in rows if r["ok"]]
        if rows:
            summary = {"max_accepted_content_tokens": max((r['content_tokens'] for r in accepted), default=0),
                       "max_all_checks_content_tokens": max((r['content_tokens'] for r in successful), default=0)}
            report["context_summary"][name] = summary
            print(f"{name}: mayor contenido aceptado={summary['max_accepted_content_tokens']}; "
                  f"mayor ensayo con todos los aciertos={summary['max_all_checks_content_tokens']}")
    print("El máximo observado es un límite inferior, no la capacidad completa. "
          "Los aciertos no equivalen a un porcentaje de inteligencia o memoria utilizada.")
    diagnostics = []
    tests = report["tests"]
    for name in routes:
        cases = {r["case"]: r for r in tests if r["route"] == name and r["test"] == "tools"}
        if cases.get("auto_none", {}).get("ok") and not cases.get("auto_default", {}).get("skipped") and not cases.get("auto_default", {}).get("ok", True):
            diagnostics.append(f"{name}: herramientas funcionan con reasoning_effort=none y fallan con el valor por defecto. "
                               "Revisar el esfuerzo de razonamiento enviado por el cliente/gateway.")
        if cases.get("auto_none", {}).get("ok") and not cases.get("auto_none_stream", {}).get("skipped") and not cases.get("auto_none_stream", {}).get("ok", True):
            diagnostics.append(f"{name}: falla el caso streaming. Revisar SSE, fragmentos de tool_calls y stream_options.")
    if set(routes) == {"gateway", "azure"}:
        gateway = {(r["test"], r.get("target"), r.get("trial"), r.get("case")): r for r in tests if r["route"] == "gateway"}
        for direct in (r for r in tests if r["route"] == "azure"):
            key = (direct["test"], direct.get("target"), direct.get("trial"), direct.get("case"))
            proxy = gateway.get(key)
            if proxy and not proxy.get("skipped") and direct["ok"] and not proxy["ok"]:
                diagnostics.append(f"Azure pasa y gateway falla: {key}. Investigar límites, reenvío de parámetros, "
                                   "transformación de mensajes o ruta del gateway; no prueba por sí solo un recorte.")
    if not diagnostics:
        diagnostics.append("Sin diferencia concluyente con estos ensayos. Revisar resultados individuales; "
                           "si ambos pasan y Copilot falla, comparar la petición real y la configuración del cliente.")
    report["diagnostics"] = list(dict.fromkeys(diagnostics))
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for finding in report["diagnostics"]:
        print(finding)
    return 0 if all(row["ok"] for row in tests) else 1


if __name__ == "__main__":
    sys.exit(main())
