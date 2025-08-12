```bash
# conda create --name rerank python=3.10
pip install uv 
uv venv rerank --python 3.10 && source rerank/bin/activate && uv pip install --upgrade pip
# uv pip install -r requirements.txt
uv pip install FlagEmbedding==1.1.2


export HF_ENDPOINT=https://hf-mirror.com
pip install huggingface-hub

# huggingface-cli download BAAI/bge-base-zh --local-dir ./bge-base-zh
huggingface-cli download BAAI/bge-reranker-base --local-dir ./bge-reranker-base
huggingface-cli download BAAI/bge-reranker-large --local-dir ./bge-reranker-large

huggingface-cli download Qwen/Qwen3-Reranker-0.6B  --local-dir ./Qwen3-Reranker-0.6B
```

# AutoDL 
```bash
source /etc/network_turbo

# conda 镜像
vim ~/.condarc
# channels:
#   - defaults
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
# show_channel_urls: true

conda create --name rerank python==3.10 -y
conda activate rerank
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia


curl -X POST "http://localhost:6006/v1/rerank" \
  -H "Authorization: Bearer ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是机器学习",
    "documents": [
      "机器学习是人工智能的一个分支",
      "深度学习是机器学习的子集",
      "计算机视觉是AI的应用领域"
    ]
  }'
```

```json
{
    "results": [
        {
            "index": 0,
            "relevance_score": 0.9996873022037057
        },
        {
            "index": 1,
            "relevance_score": 0.8071257145386546
        },
        {
            "index": 2,
            "relevance_score": 0.18769432278005577
        }
    ]
}
```

# 配置git密钥
```bash
ls -la ~/.ssh

ssh-keygen -t ed25519 -C "lizhongv@gmail.com"
cat ~/.ssh/id_ed25519.pub

ssh -T git@github.com
```