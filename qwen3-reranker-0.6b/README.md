
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