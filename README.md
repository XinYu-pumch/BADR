# MCP tools: BADR
## Bulk Academic Deep Research (BADR)散装学术深度搜索

✅迅速生成高质量生物医学综述

✅快速扩充学术论点

✅同时利用文献的摘要和原文信息

✅MCP形式调用，操作简便，无需单独的程序

✅文献文本及向量数据库完全本地化


## 实现原理：
* 使用MCP方式，从pubmed搜索指定关键词，按照相关性或时间顺序获取文献
* 使用sci-hub或PMC，获得文献的原文pdf（若无，则用摘要替代）
* 使用marker将pdf转化为markdown格式，并清理（如去除引文等）
* 基于文本段落，将文献原文分块，存储在本地ChromaDb向量数据库
* 基于所有文献的摘要搭建分层分点的综述框架
* 利用RAG方法，以综述框架索引文献文本块，填充综述框架，形成综述

  ![Uploading LLM.png…]()


## 准备工具：
* 支持MCP服务的AI客户端
  * 首选chatwise pro（付费版），本工具最完美的客户端 https://chatwise.app
  * 其次推荐chatmcp和langflow
  * **暂不支持cherry studio**（因为有mcp调阅时间限制）
* LLM
*  首选**gemini 2.5 pro**，支持的上下文长度足够，且生成的内容较有深度；建议使用GCP-vertex版本（通常截断率低），若无渠道，可使用国内号商（推荐：https://api.shubiaobiao.com   或者 https://poloai.top
*  备选：doubao-1.6（非thinking版），配置方法可参考cherry studio教程：https://docs.cherry-ai.com/websearch/volcengine
* 硅基流动（siliconflow.cn）的api（调取嵌入模型bge-m3）
* （非必须）ncbi的api和邮箱账号（在账号-设置中）


## 安装方法
```
git clone https://github.com/XinYu-pumch/BADR.git
cd BADR
pip install requirements.txt
```
chatwise中安装方法（MacOS）

设置-工具-命令处填写：
```
/your_path/python /your_path/literature_search_mcp_server_pro_2.0.py
```



## 代码运行前修改（literature_search_mcp_server_pro_2.0.py文件）

可以直接ctrl+F搜索”#设置“，为所有待改的部分

指定ChromaDB存储路径（存储你检索到的文献原文的嵌入向量）
```
# ChromaDB
CHROMA_DB_PATH = "xxxxxxxxxx" # xxxx修为你的chromadb的临时路径
```
配置硅基流动的api_key（用来调用硅基流动的嵌入模型），参考https://docs.siliconflow.cn/cn/userguide/introduction
```
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "xxxx") # 请替换为你的真实 Key 或设置环境变量
```
可选择增加NCBI_EMAIL和NCBI_API_KEY，减少调用pubmed时被ban的风险
```
NCBI_EMAIL = "your.email@example.com" # 替换成你的邮箱
NCBI_API_KEY = None # 可选
```
设置marker的工作效率
```
MARKER_WORKERS = 4 #设置 marker 使用的 worker 数量，如果电脑性能差建议选择为1
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
* 利用marker将pdf转化为md，并进行初步清洗（去除引文）
* 利用硅基流动的bge-m3嵌入模型，对原文及摘要作文本分块（基于段落，若段落过长，则再基于句号），然后存储到本地的ChromaDb中
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
以chatwise为例，模型选择gemini2.5 pro，temperature可以适度调低（如0.1），max tokens选择 65535

**第一步，下载文献并向量化**

提示词示例：
```
使用search_literature工具， pubmed上检索IL-33和冠心病，关键词用英文形式，然后获取最相关的前30篇文章，将相关信息保存入本地数据库，集合名称为0616_IL33 
```
![步骤一](https://github.com/user-attachments/assets/b5475bdd-7d77-470d-9f11-3e1b0c82733c)

**第二步，生成综述框架。该步骤非常考验LLM能力，且决定了后续综述的质量**

提示词示例：
```
综述框架写完后，使用search_text_from_chromadb工具，根据一级标题，将综述框架分为几个片段，然后分批次检索综述框架片段，每批都检索一级标题及其所属的子标题，在前文的ChromaDB 集合中搜索相关内容，用“\n”符号分割论点（请务必确保 delimiter 参数的值严格为 “\n” (单个换行符)，绝不能是 “\\n” (转义的换行符)），每一个论点检索7个文本块，然后基于基于检索的综述内容片段、所有检索到的文本块、文本块的Source Details，撰写综述片段。综述片段的要求：
- 以中文的形式； 
- 每个点尽可能清晰、详尽地阐释，每个片段大概4000-6000字；
 - 正文中要有引文，引文信息为文本块来源信息（Source Details）中指定的PMID，直接在正文中将对应的pmid插入即可，无需使用尾注的形式； 
- 前一个综述框架片段的检索后，继续自动检索下一个综述框架片段；
- 使用统一的分级标签；
- 确保所有的综述框架片段都被执行，不要有遗漏 ；
请严格执行上述要求，不要遗漏。 
```
![摘要](https://github.com/user-attachments/assets/3c0d7a9a-7ba8-4945-9cf1-cc5fba7da9a9)


**第三步，利用综述框架填充内容。注：该步骤非常消耗tokens
3.1 若想要内容更有深度，可以对综述框架分批检索——该步骤同样考验LLM能力，亲测只有gemini 2.5pro和doubao-1.6可以实现，其余flag-LLM（gpt4.1、deepseek-R1、grok3）都失败**


提示词示例：
```
综述框架写完后，使用search_text_from_chromadb工具，根据一级标题，将综述框架分为几个片段，然后分批次检索综述框架片段，每批都检索一级标题及其所属的子标题，在前文的ChromaDB 集合中搜索相关内容，用“\n”符号分割论点（请务必确保 delimiter 参数的值严格为 “\n” (单个换行符)，绝不能是 “\\n” (转义的换行符)），每一个论点检索7个文本块，然后基于基于检索的综述内容片段、所有检索到的文本块、文本块的Source Details，撰写综述片段。综述片段的要求：
- 以中文的形式； 
- 每个点尽可能清晰、详尽地阐释，每个片段大概4000-6000字；
 - 正文中要有引文，引文信息为文本块来源信息（Source Details）中指定的PMID，直接在正文中将对应的pmid插入即可，无需使用尾注的形式； 
- 前一个综述框架片段的检索后，继续自动检索下一个综述框架片段；
- 使用统一的分级标签；
- 确保所有的综述框架片段都被执行，不要有遗漏 ；
请严格执行上述要求，不要遗漏。 
```
![综述片段](https://github.com/user-attachments/assets/df3240d5-c872-4a52-9278-977903c230ff)

然后整合为一篇完整的综述。可以适当添加一些提示词完善综述。

提示词：
```
接下来，将上文的中文综述片段整合成一篇完整的文章的综述。请调取前文中完整的中文综述片段，不要有内容的遗漏，或擅自概括。另外综述使用同一的标题分级格式，如一级标题可以用中文数字（一、二、三），二级标题为阿拉伯数字（比如在第三部分中，第一点为3.1），三级标题也是阿拉伯数字（比如3.1下的第一点，为3.1.1） 
```
![完整版](https://github.com/user-attachments/assets/ecd63a47-ec9a-41c8-8703-22f30abf250d)


**3.2 若只是想快速写综述，可以直接检索整个综述框架**

提示词
```
综述框架写完后，使用search_text_from_chromadb工具，检索上一步的综述框架，在前文的ChromaDB 集合中搜索相关内容，综述框架用"\n"符号分割论点（请务必确保 delimiter 参数的值严格为 "\n" (单个换行符)，绝不能是 "\\n" (转义的换行符)），每一个论点检索7个文本块。然后基于所有检索到的文本块、文本块的Source Details和上一步的综述框架，完成一篇完整的综述。
综述的要求：
-以中文的形式；
- 每个点尽可能清晰、详尽地阐释； 
- 综述正文部分大约12000字左右。 
- 尽可能使用学术语言；
- 正文中要有引文，引文信息为文本块来源信息（Source Details）中指定的PMID，直接在正文中将对应的pmid插入即可，无需使用尾注的形式；
- 请严格执行上述要求，不要遗漏
```
![综述II](https://github.com/user-attachments/assets/05e49627-32ab-44f8-a37e-353ba8867360)

**四、生成引文（该步骤非常慢，且不一定有必要用AI操作，因为可以手动修改）**

![引文版](https://github.com/user-attachments/assets/1243655d-68ca-4f90-b3e9-82248c101a3b)
















