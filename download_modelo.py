from sentence_transformers import SentenceTransformer
import os

print("Baixando modelo para cache local...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

cache = os.path.expanduser("~/.cache/huggingface")
print(f"Modelo salvo em: {cache}")
print("A partir de agora o app funciona sem internet.")

test = model.encode("teste de funcionamento offline").tolist()
print(f"Teste OK — vetor de {len(test)} dimensões gerado.")
