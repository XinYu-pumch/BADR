# BADR
Bulk Academic Deep Research (BADR)散装学术深度搜索，帮助你快速撰写一篇可靠的生物/医学领域综述！

## 实现原理：
* 使用MCP方式，从pubmed搜索指定关键词，按照相关性或时间顺序获取指定篇数的条目（默认为20）
* 使用sci-hub或PMC，获得文献的原文；如果没有原文则以摘要替代
* 将文献原文以临时ChromaDb的形式存储在本地（此处默认使用硅基流动的bge-m3嵌入模型切割文本）
* 基于所有文献的摘要，撰写综述框架
* 利用RAG方法，根据综述框架检索并填充原文，形成综述

## 准备工具：
* 支持MCP服务的AI客户端
  * 推荐chatmcp （https://github.com/daodao97/chatmcp）
  * langflow
  * 目前暂不支持cherry studio——cherry studio默认的mcp调阅时间限制为60s，无法执行本MCP
* 支持至少100k上下文的LLM（越长越好）
  * 推荐gemini 2.5 pro，支持的上下文长度足够，且生成的内容较有深度
  * 其余推荐grok3
* 轨迹流动（siliconflow.cn）的api
* （可选）ncbi的api和邮箱账号


## 安装方法
```
git clone https://github.com/XinYu-pumch/BADR.git
cd BADR
pip install requirements.txt
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


## 代码运行前修改（编辑literature_search_mcp_server_final_decsion.py）

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

## MCP工具简介
主要包含了三个工具函数
### search_literature
输入参数包括
* keyword、db_name
* min_date（默认为空）、max_date（默认为空）
* sort_by（默认为相关性）
* num_results（返还的结果数量，默认为20）

在pubmed检索关键词，根据日期或相关性获得前几个（如前20个）结果，然后使用sci-hub或PMC获得文献的原文（如果没有原文则以摘要替代将文献原文），用轨迹流动的bge-m3嵌入模型切割文献的原文文本，并存储在本地的ChromaDb中（其中所有文献的摘要也单独保存一份），每个词条的metadata信息包括了标题和doi（用来后续作为引文格式）。
ChromaDb集合名称由用户命名
 LLM中调用方式：
<img width="1188" alt="截屏2025-04-26 01 14 08" src="https://github.com/user-attachments/assets/99bcd8b5-fdea-4e63-9f63-e177321694d5" />
如果运行提示如下报错：
<img width="1191" alt="截屏2025-04-26 01 09 40" src="https://github.com/user-attachments/assets/330f803e-691e-4be7-b68a-28bcbf53c81c" />
则可以到为literature_search_mcp_server_final_decsion.py的同层级文件夹中查看运行日志，如果日志显示如：
<img width="1132" alt="截屏2025-04-26 01 17 24" src="https://github.com/user-attachments/assets/8ef126ef-54e8-4e1c-80cd-765019d45486" />
则可以忽略该报错，说明原文已经成功保存

### get_combined_abstracts
输入参数包括db_name，功能是从指定ChromaDb集合中获取所有文献条目的摘要合集（用来传递给LLM撰写综述框架）


### search_text_from_chromadb
输入的参数包括
* db_name: 指定的ChromaDb集合
* reference_text:参考文本
* n_results：获取与分割后的参考文本的相似文本的数量，默认为5
* delimiter 参考文本的分隔符

LLM根据文献的摘要集合撰写一个综述框架，包括了很多分论点，分论点以“\n”分隔。
search_text_from_chromadb工具将根据分隔符（\n）将综述框架分割为分论点，再利用RAG的向量相似性方法，检索原文数据库中和每个分论点相似的文本，默认为5条，然后把这些文本传递给LLM
LLM最终基于综述框架和分论点的相似文本撰写综述。

***
# 使用案例
工具：chatmcp

LLM：gemini 2.5 pro preview 03-25（建议采样温度调整为0）

系统提示词(system prompt)

```
你是一个生命科学领域的专家，你将接收一个待研究主题和数据库名称，之后进行一些操作。注意！在执行每一步前，先检查一下你有的函数工具，然后在执行命令的时候使用它们.

第一步，使用search_literature工具，在pubmed上检索，关键词用英文形式，然后获取最相关的前30篇文章，将相关信息保存入本地数据库，数据名称为用户输入给你的。Check all the tools you have, and try to use them when executing my command

当执行完这一步，你需要询问请求者是否执行下面的步骤，并且完整地复述下一步的内容，不要遗漏，然后问综述的主题是什么。

第二步，从前一步保存的chromaDB集合（名称为前一步存储的名称）中获取预存的所有摘要的合并文本（存储在 Metadata 中），然后以此写一篇综述框架，主题为用户输入的主题。要求尽可能分点，涵盖的维度尽可能详实，然后将这个综述框架返还为一个文本，其中每个分点都用特殊的符号“\n”分隔开。综述框架以英文形式呈现。Check all the tools you have, and try to use them when executing my command


当执行完这一步，你需要询问请求者是否执行下面的步骤，并且完整地复述下一步的内容，不要遗漏。

第三步，根据上一步的综述框架，从前一步保存的chromaDB集合（名称为第一步存储的名称）中搜索相关内容以扩充该文本。然后基于检索到的内容和上一步的综述框架，完成一篇完整的综述，以中文的形式。另外要有引文格式，用尾注的形式，内容包括标题和doi（检索到的内容的metadata应该包含了这些信息），注意尾注要合并完全重复的项目。引文还是保持英文。Check all the tools you have, and try to use them when executing my command。
```
成果展示
![ChatMcp-IL-33 冠心病搜索](https://github.com/user-attachments/assets/958e53c2-389f-4dfc-8fd5-9d269b74eecb)


成果对比
秘塔搜索：
![IL-33在冠心病中](https://github.com/user-attachments/assets/5694cd08-d5cc-4d3e-a490-d7d00ec6fd94)

perplexity学术搜索：
![IL-33在冠心病中的作用：从分子机制到临床意义](https://github.com/user-attachments/assets/e815541f-e46a-4214-9c26-4ebfca4e855c)










