FROM python:3.11-slim

WORKDIR /app

COPY actian-vectorAI-db-beta/actian_vectorai-0.1.0b2-py3-none-any.whl .

RUN pip install actian_vectorai-0.1.0b2-py3-none-any.whl
RUN pip install flask sentence-transformers

COPY . .

RUN python download_modelo.py

EXPOSE 5000

CMD ["python", "app.py"]
