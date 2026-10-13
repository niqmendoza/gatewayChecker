# gatewayChecker

Un script pequeño para comparar **LiteLLM → Azure** con **Azure directo**, usando Chat Completions. Prueba cuánto texto acepta cada ruta, si recupera datos repartidos en ese texto y si completa el protocolo de herramientas. No modifica el gateway ni ejecuta código o comandos propuestos por el modelo.

## Instalación

Python **3.10 o posterior**:

```bash
git clone https://github.com/niqmendoza/gatewayChecker.git
cd gatewayChecker
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

En Windows, activar con `.venv\Scripts\activate`. Editá `.env` localmente: el script lo carga automáticamente **junto a `probe_llm.py`**, aunque lo ejecutes desde otra carpeta. Las variables de entorno externas prevalecen. No hay que ejecutar `source .env`; no se ejecuta su contenido ni se interpolan variables.

## LiteLLM: sólo URL y API key

Para la ruta predeterminada, completá estas dos variables en `.env`:

```dotenv
GATEWAY_URL=https://litellm.example.com/v1
GATEWAY_KEY=tu-clave-local
```

También acepta la URL raíz (`https://litellm.example.com`), un prefijo de proxy (`https://host/pasarela/v1`) o la URL completa terminada en `/chat/completions`. Conserva el prefijo y construye `/models` y `/chat/completions` sin duplicar `/v1`. Usa `Authorization: Bearer …` por defecto.

El script consulta `/models` autenticado y selecciona el ID si identifica **un único GPT-5.6 Sol** por su nombre, incluidos prefijos y sufijos de versión. Si hay varios candidatos, un alias opaco o ninguno, muestra los IDs y pide una selección explícita; no elige otro modelo silenciosamente:

```bash
python probe_llm.py --gateway-model alias-exacto --mode tools
```

También podés guardar `GATEWAY_MODEL=alias-exacto` en `.env`. La selección explícita omite `/models`, útil si tu clave no tiene permiso para listar modelos. El nombre publicado en `/models` no certifica cuál es el despliegue subyacente: comprobá el mapeo de LiteLLM.

```bash
# Herramientas: hasta ocho llamadas cortas y una consulta de modelos.
python probe_llm.py --mode tools --out reports/gateway-tools.json

# Contexto: 2k, 8k y 32k tokens aproximados; tres ensayos por tamaño.
python probe_llm.py --mode context --out reports/gateway-context.json

# Ambas familias de pruebas, solamente por LiteLLM.
python probe_llm.py
```

`--route gateway` es el valor predeterminado y no requiere datos de Azure. Si tu pasarela requiere `api-key` en lugar de Bearer, configurá `GATEWAY_AUTH=api-key`.

## Azure directo y comparación

Sólo para `--route azure` o `--route both`, agregá:

```dotenv
AZURE_URL=https://recurso.openai.azure.com/openai/v1
AZURE_KEY=tu-clave-local-de-azure
AZURE_MODEL=nombre-del-despliegue
```

`AZURE_MODEL` es el **nombre del despliegue**, que puede diferir del ID de LiteLLM. También se puede pasar con `--azure-model`. La autenticación predeterminada es `api-key`; para Entra ID usá `AZURE_AUTH=bearer` y un token vigente en `AZURE_KEY`.

Acepta la URL completa v1 o la API anterior:

```dotenv
AZURE_URL=https://recurso.openai.azure.com/openai/deployments/nombre-del-despliegue/chat/completions?api-version=VERSION-QUE-USAS
```

Se conserva la query `api-version`. Usá una versión compatible con tu despliegue. La ruta Azure no necesita configuración de LiteLLM.

```bash
python probe_llm.py --route azure --mode tools --out reports/azure-tools.json
python probe_llm.py --route both --mode all --out reports/comparacion.json
```

Para que la comparación tenga sentido, ambas rutas deben apuntar al mismo modelo y versión. Envía los mismos textos y parámetros por ensayo, salvo el nombre del modelo/despliegue. En herramientas usa el mismo recibo aleatorio entre rutas; cada respuesta proporciona su propio `tool_call_id`.

## Contexto grande, de forma explícita

El valor de referencia de entrada es **922.000 tokens**, configurable con `--input-limit`. Azure documenta GPT-5.6 Sol con contexto total de 1.050.000, entrada máxima de 922.000 y salida máxima de 128.000; esas cifras son de referencia, no una medición de tu despliegue. [Documentación de Azure](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure?pivots=azure-openai). OpenAI documenta el mismo contexto total y máximo de salida. [Modelo GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol).

Antes de consumir tokens, podés generar y contar textos sin credenciales ni llamadas de inferencia:

```bash
python probe_llm.py --dry-run --mode context --sizes 2048,8192,32768 --repeats 1
```

Para aproximarte al límite, solicitá tamaños grandes explícitamente. Este ejemplo consume varios millones de tokens de entrada entre ambas rutas; elegí tamaños según tu presupuesto y cuota:

```bash
python probe_llm.py --route both --mode context \
  --sizes 131072,262144,524288,900000 --repeats 1 \
  --timeout 600 --out reports/contexto-grande.json
```

Repetí cerca del cambio de comportamiento y usá tamaños intermedios para acotar el umbral. Para probar un rechazo por encima de la entrada de referencia, podés agregar `940000`. `--repeats` afecta el barrido de contexto; los cuatro casos de herramientas se ejecutan una vez cada uno.

Los tamaños son objetivos aproximados de **contenido**, no el total enviado. `content_tokens` se cuenta con tiktoken y excluye el envoltorio de mensajes y esquemas de herramientas. Si tiktoken no reconoce GPT-5.6 Sol, usá `--encoding o200k_base`, dejando explícita la aproximación. Si elegís otro modelo, indicá su tokenizer. `usage.prompt_tokens` es lo que reporta el servicio y no una inspección independiente de su entrada. `--output-tokens` fija el presupuesto de salida, por defecto 2048. `--input-limit` sólo cambia la referencia del informe: no amplía ningún límite.

## Qué valida y cómo leer el informe

| Prueba | Comprobación |
|---|---|
| Contexto | Recupera valores aleatorios ubicados aproximadamente al 5%, 50% y 95% del texto y combina inicio y final. |
| `auto_none` | Llamada automática estructurada con `reasoning_effort=none`. |
| `forced_none` | Selección forzada de la función `add`, con razonamiento desactivado. |
| `auto_default` | Llamada automática con el esfuerzo de razonamiento omitido. |
| `auto_none_stream` | Ciclo completo en SSE, con argumentos, nombre e ID fragmentados, `finish_reason` y `[DONE]`. |

Cada caso de herramientas valida exactamente una llamada `add` con ID y argumentos válidos, envía una suma local conocida en un mensaje `tool` y exige que el modelo copie tanto la suma como un recibo aleatorio. No ejecuta otras herramientas.

El JSON incluye modelos seleccionados/reportados, tokenizer, tokens locales/reportados, diferencia de conteo, porcentaje ocupado del límite de entrada de referencia, finalización, aciertos, latencia, etapa de fallo, resúmenes de contexto y diagnósticos comparativos. Se guarda después de cada ensayo ejecutado y al terminar; por defecto en `reports/probe_results.json`, relativo al directorio desde donde corrés el comando. `--dry-run` también guarda un informe de conteos.

Los códigos de salida son **0** si todos los ensayos pasan (o termina el dry-run), **1** si alguno falla o se omite por un error previo y **2** para argumentos/configuración/descubrimiento inválidos. `auto_default` es un ensayo de compatibilidad y puede fallar en una configuración válida: Microsoft indica que GPT-5.6 con herramientas en Chat Completions requiere `reasoning_effort=none` y recomienda Responses para herramientas con razonamiento. [Restricción documentada](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure?pivots=azure-openai).

- **Azure pasa y LiteLLM falla:** revisar límites del proxy, reenvío de parámetros, transformación de mensajes y mapeo de modelo. La diferencia por sí sola no prueba truncamiento.
- **Ambas rutas rechazan:** revisar versión, presupuesto de entrada/salida y protocolo. HTTP 413 se registra como límite del cuerpo; 429 como cuota; un timeout como transporte. Ninguno demuestra un límite de contexto. Sólo mensajes reconocidos en HTTP 400/422 se clasifican como `context_limit`.
- **Acepta texto pero pierde valores:** hay un fallo de recuperación en esa tarea; repetir y revisar logs del gateway para distinguir modelo y transformación/recorte.
- **Herramientas pasan con `none`, fallan por defecto:** revisar el esfuerzo de razonamiento enviado por el cliente o gateway.
- **SSE falla:** revisar el tipo de contenido, fragmentos de `tool_calls`, eventos de finalización y `stream_options`.
- **Todo pasa aquí pero Copilot falla:** comparar la petición real, protocolo y capacidades declaradas en VS Code. Este script verifica Chat Completions; no certifica Responses, herramientas paralelas, visión, todos los esquemas ni toda la integración de Copilot.

El máximo aceptado observado es un **límite inferior**; el mayor ensayo con todos los aciertos mide recuperación en esta prueba. No demuestra atención interna ni un porcentaje universal de “modelo aprovechado”. El porcentaje del informe mide tamaño de entrada relativo a la referencia.

El script detiene sólo la ruta afectada ante autenticación, endpoint, cuota, HTTP 5xx o transporte; registra los ensayos restantes como omitidos y sigue la otra ruta. No reintenta ni sigue redirects. `--timeout` limita espera de conexión/lectura, no el tiempo total de un ensayo. No guarda claves, URLs ni cuerpos crudos de error; consultá los logs del servicio para el detalle. Los informes sí incluyen IDs de modelo y consumo: revisalos antes de compartirlos. `.gitignore` excluye `.env`, variantes, entornos virtuales e informes JSON; `.env.example` no contiene secretos.

VS Code ofrece **Custom Endpoint** con Chat Completions, Responses y Messages. Para comparar con el script elegí Chat Completions. Declarar `toolCalling`, `maxInputTokens` o `maxOutputTokens` comunica capacidades al cliente; aumentar esos valores no amplía el modelo ni el gateway. [Configuración oficial de VS Code](https://code.visualstudio.com/docs/agent-customization/language-models).

## Pruebas locales

```bash
python -m unittest discover -s tests -v
```

La suite usa servidores HTTP simulados en localhost y claves ficticias. Cubre descubrimiento/autenticación, selección ambigua, carga de `.env` y prioridad del entorno, URLs v1/legacy, rutas separadas, peticiones equivalentes, recuperación y rechazo de contexto, errores HTTP/timeout, herramientas y recibo, SSE fragmentado/incompleto, dry-run y exclusión de secretos/informes. No usa credenciales reales ni invoca Azure o LiteLLM externos.

**Estado de validación:** probado localmente con servidores simulados y generación de contexto. No se ha verificado un despliegue real de Azure/LiteLLM: se necesitan las credenciales del usuario en su `.env` para hacerlo. Pasar estas pruebas locales no certifica ese despliegue.
