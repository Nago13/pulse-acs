from actian_vectorai import VectorAIClient, DistanceMetric, Filter, Field
from sentence_transformers import SentenceTransformer
import json, uuid
from datetime import datetime
from pathlib import Path

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = VectorAIClient("localhost:50051")
client.connect()

COLECAO = "conhecimento_clinico"
DATA_DIR = Path("C:/Users/pedro/OneDrive/Documentos/estudos/Claude/pulse/data")
DATA_DIR.mkdir(exist_ok=True)


def iniciar_visita(paciente: dict, observacao_inicial: str) -> dict:
    vetor = model.encode(observacao_inicial).tolist()

    filtro = Filter(must=[Field("regioes").any_of([paciente["estado"], "todos"])])
    resultados = client.points.search(COLECAO, vetor, limit=3, filter=filtro)

    if not resultados:
        resultados = client.points.search(COLECAO, vetor, limit=3)

    hipoteses = []
    hipoteses_detalhadas = []
    perguntas_vistas = set()
    proximas_perguntas = []

    for r in resultados:
        payload = r.payload
        hipoteses.append(payload["nome"])
        hipoteses_detalhadas.append({
            "nome": payload["nome"],
            "fonte": payload.get("fonte", ""),
            "validado_por": payload.get("validado_por"),
        })
        for pergunta in payload.get("perguntas_chave", []):
            if pergunta not in perguntas_vistas and len(proximas_perguntas) < 3:
                proximas_perguntas.append(pergunta)
                perguntas_vistas.add(pergunta)

    return {
        "id_visita": str(uuid.uuid4()),
        "hipoteses": hipoteses,
        "hipoteses_detalhadas": hipoteses_detalhadas,
        "proximas_perguntas": proximas_perguntas,
        "observacao_inicial": observacao_inicial,
    }


def atualizar_visita(
    id_visita: str,
    observacao_inicial: str,
    respostas_anteriores: list,
    paciente: dict,
) -> dict:
    contexto_completo = observacao_inicial + " " + " ".join(
        f"{r['pergunta']}: {r['resposta']}." for r in respostas_anteriores
    )

    vetor = model.encode(contexto_completo).tolist()
    resultados = client.points.search(COLECAO, vetor, limit=3)

    perguntas_ja_feitas = {r["pergunta"] for r in respostas_anteriores}

    hipoteses = []
    hipoteses_detalhadas = []
    condutas_sugeridas = []
    nivel_urgencia_sugerido = 0
    perguntas_vistas = set()
    proximas_perguntas = []

    for r in resultados:
        payload = r.payload
        hipoteses.append(payload["nome"])
        hipoteses_detalhadas.append({
            "nome": payload["nome"],
            "fonte": payload.get("fonte", ""),
            "validado_por": payload.get("validado_por"),
        })
        condutas_sugeridas.append(payload["conduta_sugerida"])
        nivel_urgencia_sugerido = max(nivel_urgencia_sugerido, payload.get("urgencia", 1))
        for pergunta in payload.get("perguntas_chave", []):
            if (
                pergunta not in perguntas_ja_feitas
                and pergunta not in perguntas_vistas
                and len(proximas_perguntas) < 3
            ):
                proximas_perguntas.append(pergunta)
                perguntas_vistas.add(pergunta)

    return {
        "hipoteses": hipoteses,
        "hipoteses_detalhadas": hipoteses_detalhadas,
        "condutas_sugeridas": condutas_sugeridas,
        "nivel_urgencia_sugerido": nivel_urgencia_sugerido,
        "proximas_perguntas": proximas_perguntas,
    }


def fechar_visita(dados_visita: dict) -> dict:
    dados_visita["sincronizado"] = False
    dados_visita["data_hora"] = datetime.now().isoformat()

    caminho = DATA_DIR / f"{dados_visita['id_visita']}.json"
    caminho.write_text(json.dumps(dados_visita, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"status": "salvo", "caminho": str(caminho)}
