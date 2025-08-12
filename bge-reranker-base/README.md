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

## 交互式构建和运行
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


  
```


# 4. 测试API
curl -X POST \
  -H "Authorization: sk-xxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is AI?",
    "documents": [
      "Artificial intelligence is the simulation of human intelligence by machines.",
      "AI is a technology that enables machines to think like humans."
    ]
  }' \
  http://localhost:6006/v1/rerank


# 预期响应
{
  "results": [
    {"index": 0, "relevance_score": 0.95},
    {"index": 1, "relevance_score": 0.87}
  ]
}

# https://github.com/labring/FastGPT/blob/main/plugins/model/rerank-bge/bge-reranker-v2-m3/requirements.txt

# 
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# 清华源
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu117

# 阿里云
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  -i https://mirrors.aliyun.com/pypi/simple \
  --extra-index-url https://download.pytorch.org/whl/cu117