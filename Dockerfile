# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# #

# # please download the model from https://huggingface.co/BAAI/bge-reranker-base and put it in the same directory as Dockerfile
# COPY ./bge-reranker-base ./bge-reranker-base

# COPY requirements.txt .

# RUN python3 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# COPY app.py Dockerfile ./

# # ENTRYPOINT python3 app.py
# ENTRYPOINT ["python3", "app.py"]


FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime
WORKDIR /reranks

COPY ./bge-reranker-base ./bge-reranker-base

COPY requirements.txt .
# RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --default-timeout=1000 -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY bge_app2.py .

EXPOSE 6006

CMD ["uvicorn", "bge_app2:app", "--host", "0.0.0.0", "--port", "6006"]
# uvicorn bge_app2:app --host 0.0.0.0 --port 8000