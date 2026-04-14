from flask import Flask, request, jsonify, render_template
from copiloto import iniciar_visita, atualizar_visita, fechar_visita
import json
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

DATA_DIR = Path("C:/Users/pedro/OneDrive/Documentos/estudos/Claude/pulse/data")
SYNC_DIR = DATA_DIR / "sincronizados"
DATA_DIR.mkdir(exist_ok=True)
SYNC_DIR.mkdir(exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/iniciar", methods=["POST"])
def iniciar():
    try:
        body = request.get_json()
        resultado = iniciar_visita(body["paciente"], body["observacao_inicial"])
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/atualizar", methods=["POST"])
def atualizar():
    try:
        body = request.get_json()
        resultado = atualizar_visita(
            body["id_visita"],
            body["observacao_inicial"],
            body["respostas_anteriores"],
            body["paciente"],
        )
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/fechar", methods=["POST"])
def fechar():
    try:
        body = request.get_json()
        resultado = fechar_visita(body)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/pendentes", methods=["GET"])
def pendentes():
    try:
        visitas = []
        for arquivo in DATA_DIR.glob("*.json"):
            dados = json.loads(arquivo.read_text(encoding="utf-8"))
            if not dados.get("sincronizado", True):
                visitas.append({
                    "id_visita": dados.get("id_visita"),
                    "nome_paciente": dados.get("paciente", {}).get("nome", ""),
                    "data_hora": dados.get("data_hora"),
                })
        return jsonify(visitas)
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/sincronizar", methods=["POST"])
def sincronizar():
    try:
        sincronizadas = 0
        for arquivo in DATA_DIR.glob("*.json"):
            dados = json.loads(arquivo.read_text(encoding="utf-8"))
            if dados.get("sincronizado", True):
                continue

            paciente = dados.get("paciente", {})
            hipoteses = dados.get("hipoteses", [])
            conduta_real = dados.get("conduta_real", "")
            urgencia = dados.get("urgencia_confirmada", "")
            data_hora = dados.get("data_hora", "")

            resumo = (
                f"VISITA PULSE — {data_hora}\n"
                f"{'=' * 40}\n"
                f"Paciente : {paciente.get('nome', '')} | "
                f"Idade: {paciente.get('idade', '')} | "
                f"Sexo: {paciente.get('sexo', '')} | "
                f"UF: {paciente.get('estado', '')}\n"
                f"Condições crônicas: {', '.join(paciente.get('condicoes_cronicas', []))}\n\n"
                f"Observação inicial: {dados.get('observacao_inicial', '')}\n\n"
                f"Hipóteses: {', '.join(hipoteses)}\n\n"
                f"Conduta registrada pelo ACS:\n{conduta_real}\n\n"
                f"Urgência confirmada: {urgencia}\n"
                f"{'=' * 40}\n"
                f"Respostas coletadas:\n"
            )
            for r in dados.get("respostas_anteriores", []):
                resumo += f"  P: {r.get('pergunta', '')}\n"
                resumo += f"  R: {r.get('resposta', '')}\n\n"

            txt_path = SYNC_DIR / f"{dados['id_visita']}.txt"
            txt_path.write_text(resumo, encoding="utf-8")

            dados["sincronizado"] = True
            dados["data_sincronizacao"] = datetime.now().isoformat()
            arquivo.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
            sincronizadas += 1

        return jsonify({"sincronizadas": sincronizadas})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
