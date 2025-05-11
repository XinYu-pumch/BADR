# MCP tools: BADR
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
* LLM：仅推荐gemini 2.5 pro，支持的上下文长度足够，且生成的内容较有深度（备注：推荐openrouter付费版本，官网版本容易截断）
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



## 使用方法
chatwise中agent，系统提示词如下：

```
你是一个生命医学专家，你将接收一个关键词、数据库名称和待研究主题，然后执行如下操作：

第一步，首先使用search_literature工具， pubmed上检索用户发送给你的关键词，关键词用英文形式检索，获取最相关的前30篇文章，将相关信息保存入本地数据库，集合名称为为用户发送给你的。


第二步，接着使用get_combined_abstracts工具，从前一步的chromadb集合中获取预存的所有摘要的合并文本（存储在Metadata中），然后以此写一篇生命科学综述的框架，主题为用户发送给你的。要求尽可能分点，涵盖的维度尽可能详实，然后将这个综述框架返还为一个文本，其中每个分点之间都要换行，分论点的结尾都以特殊的符号“\n”分结束。综述框架以英文形式呈现。

第三步，综述框架写完后，使用search_text_from_chromadb工具，检索上一步的综述框架，在前文的ChromaDB 集合中搜索相关内容，综述框架用"\n"符号分割论点（请务必确保 delimiter 参数的值严格为 "\n" (单个换行符)，绝不能是 "\\n" (转义的换行符)），每一个论点检索7个文本块。然后基于所有检索到的文本块、文本块的Source Details，以上一步的综述框架为参考，完成一篇完整的综述，并严格遵照如下要求： 1）综述以中文的形式，用学术语言的口吻； 2）逻辑清晰，减少赘述，减少重复的内容，不用完全遵循综述框架； 3）综述正文部分大约8000字左右。 写完综述后，再为综述添加引文，引文的要求： 1）引文信息使用文本块来源信息（Source Details）中指定的PMID；2）直接在正文中将对应的pmid插入即可，无需使用尾注的形式。 


 请务必严格遵照上述指令进行你的操作。
```
当写完后可以单独输入prompt完善引文信息
```
最后为该文章生成完整的引文格式：首先将前一步中完整的综述传递给get_reference工具，get_reference工具将从前文中的chromadb集合中提取文献信息，为该文章生成尾注。然后基于文献列表，将正文中的pmid替换为尾注中的序号（正文中如果一处内容对应多个引用，则按引用的序号大小重新排列）。并且为正文添加尾注，尾注严格为get_reference生成的格式化的尾注列表。该步骤无需使用sequential_thinking工具。正文内容本身不要改变。![image](https://github.com/user-attachments/assets/590a2b9a-2a5d-49ef-98c5-17a041ddd734)

```

模型选择gemini2.5 pro，temperature可以适度调低（如0.1），max tokens选择 65536

输入的提示词类似：
```
关键词：IL-33和冠心病；
保存集合名称：0504_IL33_9；
综述主题：IL-33在冠心病中的作用
```

AI将自动帮你生成综述！

参考效果：
![longshot20250505181819](https://github.com/user-attachments/assets/7d83023c-0876-4f0a-a3db-f0bffd22a743)

















