# BADR
Bulk Academic Deep Research (BADR)散装学术深度搜索，帮助你快速撰写一篇可靠的生物/医学领域综述！

实现原理：
* 使用MCP方式，从pubmed搜索指定关键词，按照相关性或时间顺序获取指定篇数的条目（默认为20）
* 使用sci-hub或PMC，获得文献的原文；如果没有原文则以摘要替代
* 将文献原文以临时ChromaDb的形式存储在本地（此处默认使用硅基流动的bge-m3嵌入模型切割文本）
* 基于所有文献的摘要，撰写综述框架
* 利用RAG方法，根据综述框架检索并填充原文，形成综述

准备工具：
* 支持MCP服务的AI客户端
  * 推荐chatmcp （https://github.com/daodao97/chatmcp）
  * langflow
  * 目前暂不支持cherry studio——cherry studio默认的mcp调阅时间限制为60s，无法执行本MCP
* 支持至少100k上下文的LLM（越长越好）
  * 推荐gemini 2.5 pro，支持的上下文长度足够，且生成的内容较有深度
  * 其余推荐grok3
* 轨迹流动（siliconflow.cn）的api
* （可选）ncbi的api和邮箱账号


安装方法
```
git clone https://github.com/XinYu-pumch/BADR.git
cd BADR
pip install r
```
chatmcp中安装方法（MacOS）

命令（为你的python的绝对路径，terminal中运行which python可查看）
```
/your_path/python
```
参数（为literature_search_mcp_server_final_decsion.py的绝对路径）
```
/your_path/literature_search_mcp_server_final_decsion.py
```

参考cursor的json形式如下：
```
<json>
{
      "name": "literature_search_mcp_server",
      "type": "stdio",
      "description": "literature_search_mcp_server",
      "isActive": true,
      "command": "/your_path/python",
      "args": [
        "/your_path/literature_search_mcp_server_final_decsion.py"
      ]
    }
 ```   


代码运行前修改（编辑literature_search_mcp_server_final_decsion.py）
指定ChromaDB存储路径（存储你检索到的文献原文的嵌入向量）
```
# ChromaDB
CHROMA_DB_PATH = "xxxxxxxxxx" # xxxx修为你的chromadb的临时路径
```
配置硅基流动的api_key（用来调用硅基流动的嵌入模型），参考https://docs.siliconflow.cn/cn/userguide/introduction
```
# 硅基流动 Embedding API
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/embeddings"
# !!! 安全警告：切勿在生产代码中硬编码 API Key !!!
# 建议使用环境变量或其他安全方式管理
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "xxxxxxxxxxxx") #把xxx替换为你的硅基流动的api_key
```
可选择增加NCBI_EMAIL和NCBI_API_KEY，减少调用pubmed时被ban的风险
```
NCBI_EMAIL = "your.email@example.com" # 替换成你的邮箱
NCBI_API_KEY = None # 可选
```








