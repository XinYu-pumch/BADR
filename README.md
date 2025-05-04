# MCO tools: BADR
## Bulk Academic Deep Research (BADR)散装学术深度搜索

✅迅速生成高质量生物医学综述

✅快速扩充学术论点

✅同时利用文献的摘要和原文信息

✅MCP形式调用，操作简便，无需单独的程序


## 实现原理：
* 使用MCP方式，从pubmed搜索指定关键词，按照相关性或时间顺序获取文献
* 使用sci-hub或PMC，获得文献的原文（若无，则用摘要替代）
* 基于语意分隔，将文献原文分块，存储在本地ChromaDb向量数据库
* 基于所有文献的摘要搭建综述框架
* 利用RAG方法，填充综述框架，形成综述

## 准备工具：
* 支持MCP服务的AI客户端
  * 首选chatwise pro，本工具最完美的客户端（https://chatwise.app/）
  * 其次推荐chatmcp和langflow
  * 暂不支持cherry studio（因为有mcp调阅时间限制）
* LLM：仅推荐gemini 2.5 pro，支持的上下文长度足够，且生成的内容较有深度
* 轨迹流动（siliconflow.cn）的api（调阅bge-m3模型）
* （可选）ncbi的api和邮箱账号


## 安装方法
```
git clone https://github.com/XinYu-pumch/BADR.git
cd BADR
pip install requirements.txt
```
chatwise中安装方法（MacOS）

设置-工具-命令处填写：
```
/your_path/python /your_path/literature_search_mcp_server_pro.py
```



## 代码运行前修改（编辑literature_search_mcp_server_pro.py）

指定ChromaDB存储路径（存储你检索到的文献原文的嵌入向量）
```
# ChromaDB
CHROMA_DB_PATH = "xxxxxxxxxx" # xxxx修为你的chromadb的临时路径
```
配置硅基流动的api_key（用来调用硅基流动的嵌入模型），参考https://docs.siliconflow.cn/cn/userguide/introduction
```
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "xxxx") # 请替换为你的真实 Key 或设置环境变量
if SILICONFLOW_API_KEY == "xxxx":
     logging.warning("Using a placeholder SiliconFlow API Key. Please set the SILICONFLOW_API_KEY environment variable or replace the placeholder.")
```
可选择增加NCBI_EMAIL和NCBI_API_KEY，减少调用pubmed时被ban的风险
```
NCBI_EMAIL = "your.email@example.com" # 替换成你的邮箱
NCBI_API_KEY = None # 可选
```

## MCP工具简介
主要包含了四个工具函数
### search_literature

#### 输入参数
* Keyword
* db_name
* min_date（默认为空）
* max_date（默认为空）
* sort_by（默认为相关性）
* num_results（返还的结果数量，默认为20）
#### 功能
* 在pubmed检索关键词，根据日期或相关性获得前几个结果，然后使用sci-hub或PMC获得文献的pdf原文（若无原文，则以摘要替代）并下载；
* 利用marker将pdf转化为md；
* 利用硅基流动的bge-m3嵌入模型，对原文及摘要作语意切割，然后存储到本地的ChromaDb中
* 将所有文献的摘要单独保存
* 将所有文献的标题、doi、pmid单独保存



### get_combined_abstracts
功能：指定ChromaDb集合中获取所有文献的摘要（用来传递给LLM撰写综述框架）

### get_referenece
功能：指定ChromaDb集合中获取所有文献条目的标题、doi、pmid（用来传递给LLM生成引文）


### search_text_from_chromadb
输入的参数包括
* db_name: 指定的ChromaDb集合
* reference_text:参考文本
* n_results：获取与分割后的参考文本的相似文本的数量，默认为5
* delimiter 参考文本的分隔符
功能：接收输入的文字，以“\n”符号分割文本，然后批量对分割后的文本利用RAG的向量相似性方法，检索原文数据库相似的文本并返还给LLM（默认为5条）













