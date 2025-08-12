```bash 
# docker pull pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 构建镜像
docker build -t rerank-api:latest .

# 启动容器
docker run -d \
  --name rerank-service \
  -p 6006:6006 \
  -e ACCESS_TOKEN="ACCESS_TOKEN" \
  rerank-api:latest

# 检查容器状态
docker ps -f name=rerank-service


# 测试API
curl -X POST \
  -H "Authorization: Bearer your_secret_token" \
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

## 其他命令
# 查看docker镜像 
docker images
# 停止容器	
docker stop rerank-service
# 重启容器	
docker restart rerank-service
# 查看日志
docker logs -f rerank-service
# 进入容器（调试）	
docker exec -it rerank-service /bin/bash
# 删除容器	
docker rm -f rerank-service




## bug出现
# 查看已停止的容器
docker ps -a -f name=rerank-service

# 查看容器日志
docker logs rerank-service

# 查看端口是否占用
netstat -tulnp | grep 6006

# 删除原有镜像 
docker rm -f rerank-service

# 以交互模式运行，方便查看输出
docker run -it --name rerank-service -p 6006:6006 -e ACCESS_TOKEN="ACCESS_TOKEN" rerank-api:latest
```

## 交互式构建和运行
以交互方式构建镜像和运行代码
```bash
docker run -it --name rerank-builder pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime /bin/bash 

# 更新包管理器（如需要安装git等）
# apt-get update && apt-get install -y git

pip install FlagEmbedding==1.2.8
pip install transformers==4.28
```