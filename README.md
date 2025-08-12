

# 配置git密钥
```bash
ls -la ~/.ssh

ssh-keygen -t ed25519 -C "lizhongv@gmail.com"
cat ~/.ssh/id_ed25519.pub

# 验证
ssh -T git@github.com
```

# 下载模型
```bash 
# source /etc/network_turbo
export HF_ENDPOINT=https://hf-mirror.com

pip install huggingface-hub

# huggingface-cli download BAAI/bge-base-zh --local-dir ./bge-base-zh
# huggingface-cli download BAAI/bge-reranker-base --local-dir ./bge-reranker-base
hf download  BAAI/bge-reranker-base --local-dir ./bge-reranker-base
hf download BAAI/bge-reranker-large --local-dir ./bge-reranker-large
hf download Qwen/Qwen3-Reranker-0.6B  --local-dir ./Qwen3-Reranker-0.6B
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir ./bge-reranker-v2-m3
hf download BAAI/bge-reranker-v2-m3 --local-dir ./bge-reranker-v2-m3 --force


# pip install modelscope
modelscope download --model BAAI/bge-reranker-v2-m3  --local_dir ./bge-reranker-v2-m3
```

# AutoDL 

```bash
### 学术加速
# source /etc/network_turbo

### 配置conda镜像
# vim ~/.condarc
# channels:
#   - defaults
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
# show_channel_urls: true

### 创建并激活虚拟环境
conda create --name rerank python==3.10 -y
conda activate rerank

### 安装pytorch
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# 清华源
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu117

# 阿里云
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
#   -i https://mirrors.aliyun.com/pypi/simple \
#   --extra-index-url https://download.pytorch.org/whl/cu117

### 验证
python # 交互式
>>> import torch 
>>> torch.version.cuda
'11.7'
###  pytorch版本过低，报错！！！

### 删除已创建的虚拟环境
conda remove -n rerank --all

### 新创建和激活虚拟环境
conda create --name rerank python==3.10 -y
conda activate rerank

# CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

### 注意安装指定版本numpy pip install numpy==1.26.4 !!!!
# 验证

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

# API 调用

```bash
curl -X POST "http://localhost:6006/v1/rerank" \
  -H "Authorization: Bearer sk-xxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是机器学习",
    "documents": [
      "计算机视觉是AI的应用领域",
      "机器学习是人工智能的一个分支",
      "深度学习是机器学习的子集"
    ]
  }'


# {
#     "results": [
#         {
#             "index": 1,
#             "relevance_score": 0.9996873022037057
#         },
#         {
#             "index": 2,
#             "relevance_score": 0.8071257145386546
#         },
#         {
#             "index": 0,
#             "relevance_score": 0.18769432278005577
#         }
#     ]
# }


curl -X POST "http://localhost:6006/v1/rerank" \
  -H "Authorization: Bearer sk-xxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "气候变化的影响",
    "documents": [
      "全球变暖导致冰川融化",
      "气候变化影响农作物产量",
      "极端天气事件增加",
      "海平面上升威胁沿海城市",
      "生物多样性减少",
      "可再生能源发展迅速"
    ]
  }'
{
    "results": [
        {
            "index": 1,
            "relevance_score": 0.8932942033273173
        },
        {
            "index": 0,
            "relevance_score": 0.4485962275080292
        },
        {
            "index": 4,
            "relevance_score": 0.28563940295861906
        },
        {
            "index": 2,
            "relevance_score": 0.2834893375579417
        },
        {
            "index": 3,
            "relevance_score": 0.07953733668665099
        },
        {
            "index": 5,
            "relevance_score": 0.00007020986676218509
        }
    ]
}
```

# 内部服务器

```bash 
# 1. 构建镜像
docker build -t rerank-api:v1 .         

# 2. 启动容器
docker run -d \
  --name rerank-api \
  -p 6006:6006 \
  -e ACCESS_TOKEN="sk-xxxx" \
  rerank-api:v1

# 3. 检查容器状态
docker ps -f name=rerank-api


### 错误排查
# 查看端口是否占用
netstat -tulnp | grep 6006
# 查看日志
docker logs -f rerank-api
# 停止容器	
docker stop rerank-api
# 重启容器	
docker restart rerank-api
# 强制删除容器	
docker rm -f rerank-api
# 查看已停止的容器
docker ps -a -f name=rerank-api
# 查看所有容器的状态
docker ps -a
# 查看正在运行的容器
docker ps



### 镜像命令
# 查看镜像 
docker images
# 强制删除镜像 
docker rmi -f rerank-api:v1
```

以交互方式构建镜像和运行代码
```bash
# 进入容器（调试）	
docker exec -it rerank-service /bin/bash
# 以交互模式运行，方便查看输出
docker run -it --name rerank-service -p 6006:6006 -e ACCESS_TOKEN="ACCESS_TOKEN" rerank-api:latest
docker run -it --name rerank-builder pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime /bin/bash 

# 更新包管理器（如需要安装git等）
# apt-get update && apt-get install -y git

pip install FlagEmbedding
pip install transformers==4.51.0 

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu118



docker pull docker.1ms.run/pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://mirrors.aliyun.com/pytorch-wheels/cu118/ 

pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu118
```




```bash
# conda create --name rerank python=3.10
pip install uv 
uv venv rerank --python 3.10 && source rerank/bin/activate && uv pip install --upgrade pip
# uv pip install -r requirements.txt
uv pip install FlagEmbedding==1.1.2


```
# 错误排查
 PyTorch 版本过低（2.0.1+cu117），而运行 FlagEmbedding 和 transformers 需要 PyTorch ≥ 2.1。此外，torch.distributed.device_mesh 模块在 PyTorch 2.0 中不存在，导致依赖它的 accelerate 库报错。

# 参考

1. https://github.com/labring/FastGPT/blob/main/plugins/model/rerank-bge/bge-reranker-v2-m3/requirements.txt
2. https://www.cnblogs.com/zxporz/p/18898538
3. https://doc.fastgpt.cn/docs/introduction/development/custom-models/bge-rerank
4. https://n40.cn/archives/508



# 方式二
## 环境配置

```bash 
conda create --name rerank python==3.10 -y
conda activate rerank2 

pip install fastapi==0.103.1
pip install uvicorn==0.23.2
pip install pydantic==2.4.2
pip install transformers==4.30.0

# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install sentence-transformers==2.3.0



# 先彻底卸载现有torch
pip uninstall torch torchvision torchaudio -y

# 清理残留文件
rm -rf ~/.cache/pip
rm -rf ~/.cache/torch

# 安装兼容的CPU版本PyTorch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu



```
## 运行app

```bash
uvicorn bge_app2:app --host 0.0.0.0 --port 8000

# curl -X GET "http://localhost:8000/health"
# curl -X GET "http://localhost:8000/model-info"

curl -X POST "http://localhost:8000/rerank" \
-H "Content-Type: application/json" \
-d '{
  "query": "气候变化的影响",
  "documents": [
    "全球变暖导致冰川融化",
    "气候变化影响农作物产量",
    "极端天气事件增加",
    "海平面上升威胁沿海城市",
    "生物多样性减少",
    "可再生能源发展迅速"
  ],
  "batch_size": 4
}'
```

## 回复格式

```json
{
    "model": "bge-reranker-base",
    "device": "cpu",
    "processing_time_seconds": 0.192,
    "documents_processed": 6,
    "results": [
        {
            "document": "气候变化影响农作物产量",
            "score": 0.8932942152023315,
            "rank": 1
        },
        {
            "document": "全球变暖导致冰川融化",
            "score": 0.448596328496933,
            "rank": 2
        },
        {
            "document": "生物多样性减少",
            "score": 0.285638689994812,
            "rank": 3
        },
        {
            "document": "极端天气事件增加",
            "score": 0.28349044919013977,
            "rank": 4
        },
        {
            "document": "海平面上升威胁沿海城市",
            "score": 0.07953739166259766,
            "rank": 5
        },
        {
            "document": "可再生能源发展迅速",
            "score": 0.00007021013152552769,
            "rank": 6
        }
    ]
}
```



## 基本调用方式
```bash
curl -X POST "http://localhost:6006/v1/rerank" \
     -H "Authorization: Bearer sk-xxxx" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "What is the capital of China?",
           "documents": [
               "The capital of China is Beijing.",
               "China is a country in East Asia.",
               "Beijing is a large city with many historical sites."
           ]
         }'
```

## 自定义instruction调用方式

```bash 
curl -X POST "http://localhost:6006/v1/rerank" \
     -H "Authorization: Bearer sk-xxxx" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "解释量子力学",
           "documents": [
               "量子力学是研究物质世界微观粒子运动规律的物理学分支。",
               "北京是中国的首都。",
               "量子纠缠是量子力学中的重要现象。"
           ],
           "instruction": "判断文档是否准确回答了查询中的专业问题"
         }'
```