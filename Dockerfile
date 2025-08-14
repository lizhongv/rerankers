FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /reranks

COPY ./bge-reranker-large ./bge-reranker-large
COPY ./bge-reranker-v2-m3 ./bge-reranker-v2-m3 

COPY requirements.txt .
RUN pip install --default-timeout=1000 -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY bge_app3.py .

EXPOSE 6006
CMD ["uvicorn", "bge_app3:app", "--host", "0.0.0.0", "--port", "6006"]
