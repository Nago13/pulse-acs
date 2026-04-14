# Pulse — Copiloto de Saúde para ACS

Sistema offline de suporte a Agentes Comunitários de Saúde (ACS) no Brasil.

## O problema
Mais de 200 mil ACS visitam comunidades rurais sem internet diariamente.
Eles tomam decisões de saúde sem nenhuma ferramenta de apoio.

## A solução
O Pulse é um copiloto — não um diagnóstico automático.
Ele guia o ACS com perguntas progressivas, sugere hipóteses e condutas,
e registra o atendimento offline para sincronizar com a UBS depois.

## Por que o Actian VectorAI DB?
Busca semântica offline: entende que "febre com tremores noturnos" e
"calafrios intermitentes febris" são a mesma coisa — sem internet,
sem nuvem, sem latência. Nenhum banco convencional faz isso.

A busca é híbrida: semântica (significado dos sintomas) + filtros
estruturados (região geográfica, faixa etária) combinados em um
único resultado ranqueado por relevância clínica.

## Como rodar

### Requisitos
- Docker Desktop rodando
- Python 3.10+
- Dependências instaladas

### Instalação
```
git clone https://github.com/hackmamba-io/actian-vectorAI-db-beta
cd actian-vectorAI-db-beta
docker compose up -d
cd ../pulse
python -m venv venv
venv\Scripts\activate
pip install ..\actian-vectorAI-db-beta\actian_vectorai-0.1.0b2-py3-none-any.whl
pip install flask sentence-transformers
python download_modelo.py
python seed_database.py
python app.py
```

Acesse: http://localhost:5000

## Fluxo do atendimento

1. ACS abre o app no celular (funciona via Wi-Fi local, sem internet)
2. Informa dados do paciente e descreve o que observa
3. Pulse sugere perguntas progressivas baseadas nos sintomas
4. A cada resposta, as hipóteses são refinadas
5. Pulse sugere conduta — ACS confirma ou ajusta
6. Atendimento salvo localmente em JSON
7. Quando chegar à UBS com internet: sincroniza tudo automaticamente

## Arquitetura técnica

- **Banco vetorial:** Actian VectorAI DB (Docker, porta 50051)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (384 dimensões, cache local)
- **Busca:** híbrida — semântica + filtros por região e faixa etária
- **Backend:** Flask (Python)
- **Interface:** HTML/CSS/JS mobile-first
- **Persistência:** JSON local → sincronização com UBS

## Base de conhecimento
25 condições clínicas prevalentes no Brasil indexadas com:
- Descrição clínica em linguagem leiga
- Perguntas-chave para triagem
- Conduta sugerida por nível de urgência
- Filtros por região geográfica e faixa etária

## Requisito técnico atendido
Busca híbrida (Hybrid Fusion): combinação de busca semântica vetorial
com filtered search por metadados estruturados (região, faixa etária),
gerando ranking unificado por relevância clínica.

## Equipe
Construído durante o Actian VectorAI DB Build Challenge — abril 2026.
