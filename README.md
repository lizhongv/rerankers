```bash
# conda create --name rerank python=3.10
pip install uv 
uv venv rerank --python 3.10 && source rerank/bin/activate && uv pip install --upgrade pip
# uv pip install -r requirements.txt
uv pip install FlagEmbedding==1.1.2


export HF_ENDPOINT=https://hf-mirror.com
pip install huggingface-hub

huggingface-cli download BAAI/bge-base-zh --local-dir ./bge-base-zh

huggingface-cli download BAAI/bge-reranker-base --local-dir ./bge-reranker-base
huggingface-cli download BAAI/bge-reranker-large --local-dir ./bge-reranker-large
```

# 配置git密钥
```bash
ls -la ~/.ssh

ssh-keygen -t ed25519 -C "lizhongv@gmail.com"
cat ~/.ssh/id_ed25519.pub

ssh -T git@github.com
```