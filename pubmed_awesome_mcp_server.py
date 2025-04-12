#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import requests
import time
import xml.etree.ElementTree as ET
import json # 用于解析 ESearch 的 JSON 响应
import re   # 需要 re 模块
import sys  # 导入 sys 以便在 FastMCP 导入失败时使用 stderr
from typing import Dict, Any, List, Optional, Tuple

# 假设 FastMCP 可以这样导入
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("错误：无法导入 FastMCP。请确保 'mcp.server' 库已安装且包含 FastMCP。", file=sys.stderr)
    # 提供一个假的 FastMCP 以便代码能解析，但在运行时会失败
    class FakeMCP:
        def __init__(self, name): pass
        def tool(self): return lambda func: func
        def run(self): print("错误：FastMCP 未找到，无法运行服务器。", file=sys.stderr)
    FastMCP = FakeMCP


# --- 配置 ---
# 建议设置您的邮箱地址和 API 密钥
NCBI_EMAIL = "your.email@example.com" # !!! 替换成你的邮箱 !!!
NCBI_API_KEY = None # "your_api_key_here" (可选)

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("PubMedAwesomeTools") # 给你的 MCP 服务命名

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# --- 辅助函数 ---

def make_request_with_retry(url: str, params: Dict[str, Any], method: str = 'get', max_retries: int = 3, wait_time: float = 1.0) -> requests.Response:
    """使用 requests 发送带有重试机制的请求"""
    effective_params = params.copy()
    if NCBI_EMAIL and NCBI_EMAIL != "your.email@example.com":
        effective_params['email'] = NCBI_EMAIL
    if NCBI_API_KEY and NCBI_API_KEY != "your_api_key_here": # 修正这里的判断条件
        effective_params['api_key'] = NCBI_API_KEY

    logger.debug(f"Making {method.upper()} request to {url} with params: {effective_params}")

    response = None # 初始化 response
    for attempt in range(max_retries):
        try:
            if method.lower() == 'get':
                response = requests.get(url, params=effective_params, timeout=30) # 增加超时
            elif method.lower() == 'post':
                 # EFetch/ESummary 对大量 IDs 可能需要 POST
                 response = requests.post(url, data=effective_params, timeout=60) # 增加超时
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status() # 如果状态码不是 2xx，则抛出 HTTPError
            logger.debug(f"Request successful (attempt {attempt + 1}/{max_retries}), Status: {response.status_code}")
            return response
        except requests.exceptions.Timeout as e:
             logger.warning(f"Request timed out (attempt {attempt + 1}/{max_retries}): {str(e)}")
             if attempt == max_retries - 1: raise
             time.sleep(wait_time)
             wait_time *= 2
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            # 对于 4xx 错误（如 400 Bad Request），通常不应重试
            # 检查 response 是否已赋值
            if response is not None and 400 <= response.status_code < 500:
                 logger.error(f"Client error ({response.status_code}), not retrying. Response: {response.text[:500]}") # 记录部分响应体
                 raise # 重新抛出异常，让调用者处理
            if attempt == max_retries - 1: raise # 如果是最后一次尝试，则抛出异常
            time.sleep(wait_time)
            wait_time *= 2 # 指数退避
    # 此处理论上不会到达，因为循环要么返回要么在最后一次尝试时 raise
    # 添加一个显式的 raise 来处理万一循环结束却没有返回或 raise 的情况
    raise Exception("Request failed after multiple retries")


def parse_pubmed_xml_details(xml_content: bytes) -> List[Dict[str, Any]]:
    """从 EFetch 返回的 PubMed XML 中解析文章详情，包括 PMCID"""
    results = []
    if not xml_content:
        return results
    try:
        root = ET.fromstring(xml_content)
        for article in root.findall('.//PubmedArticle'):
            entry = {}
            # PMID
            pmid_elem = article.find('.//PMID')
            entry['pmid'] = pmid_elem.text if pmid_elem is not None else None

            # Title
            title_elem = article.find('.//ArticleTitle')
            # 处理可能的 <i/>, <b/> 等标签，获取纯文本
            entry['title'] = "".join(title_elem.itertext()).strip() if title_elem is not None else "N/A"


            # Abstract
            abstract_texts = []
            # 查找 Abstract 节点下的所有 AbstractText 节点
            abstract_node = article.find('.//Abstract')
            if abstract_node is not None:
                 for abst_text in abstract_node.findall('./AbstractText'):
                     label = abst_text.get('Label')
                     # 获取包含子标签在内的所有文本
                     text_parts = [t for t in abst_text.itertext()]
                     text = "".join(text_parts).strip()
                     if text:
                         if label:
                             abstract_texts.append(f"{label}: {text}")
                         else:
                             abstract_texts.append(text)
            entry['abstract'] = "\n".join(abstract_texts) if abstract_texts else None


            # Journal Title
            journal_elem = article.find('.//Journal/Title')
            entry['journal'] = journal_elem.text if journal_elem is not None else None

            # --- 获取各种 Article IDs (包括 DOI 和 PMCID) ---
            doi = None
            pmcid = None # 初始化 PMCID
            article_id_list_node = article.find('.//ArticleIdList')
            if article_id_list_node is not None:
                for item in article_id_list_node.findall('./ArticleId'):
                    id_type = item.get('IdType')
                    if id_type == 'doi':
                        doi = item.text
                    elif id_type == 'pmc': # <--- 新增：查找 PMCID
                        pmcid = item.text # 获取 PMCID (通常格式如 "PMC1234567")

            # 备选方案：从 ELocationID 查找 DOI (像 enhanced server 那样)
            if not doi:
                 doi_elem = article.find(".//ELocationID[@EIdType='doi']")
                 if doi_elem is not None:
                     doi = doi_elem.text

            entry['doi'] = doi if doi else None
            entry['pmcid'] = pmcid if pmcid else None # <--- 新增：将 PMCID 添加到结果字典

            if entry.get('pmid'): # 确保至少有 pmid
                results.append(entry)

    except ET.ParseError as e:
        logger.error(f"Failed to parse PubMed XML: {e}", exc_info=True)
        # 可以选择抛出异常或返回空列表/部分结果
        raise # 或者返回已成功解析的部分： return results
    except Exception as e:
        logger.error(f"Error processing PubMed XML data: {e}", exc_info=True)
        raise

    return results

def parse_pmc_summary_xml(xml_content: bytes) -> List[Dict[str, Any]]:
    """从 ESummary 返回的 PMC XML 中解析标题和链接"""
    results = []
    if not xml_content:
        return results
    try:
        root = ET.fromstring(xml_content)
        for doc_sum in root.findall('.//DocSum'):
            entry = {}
            uid = doc_sum.find('./Id').text # 内部 UID

            # 获取标题
            title_elem = doc_sum.find("./Item[@Name='Title']")
            entry['title'] = title_elem.text if title_elem is not None and title_elem.text else "N/A"

            # 获取 PMCID
            pmcid = None
            article_ids_node = doc_sum.find("./Item[@Name='ArticleIds']")
            if article_ids_node is not None:
                for item in article_ids_node.findall("./Item[@Name='pmcid']"):
                    raw_pmcid = item.text
                    if raw_pmcid and 'PMC' in raw_pmcid:
                        # 提取 'PMC...' 部分
                        match = re.search(r'(PMC\d+(\.\d+)?)', raw_pmcid)
                        if match:
                            pmcid = match.group(1)
                            break # 找到第一个就用

            if pmcid:
                 entry['pmc_link'] = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                 results.append(entry)
            else:
                # 注意：这里可能没有 PMCID，因为我们是从 PMC 的 UID 获取摘要的
                logger.debug(f"Could not find valid PMCID in ESummary for DocSum UID {uid} with title '{entry['title']}'.")

    except ET.ParseError as e:
        logger.error(f"Failed to parse PMC ESummary XML: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error processing PMC ESummary XML data: {e}", exc_info=True)
        raise

    return results

# --- 新增：PMC 全文解析函数 ---
def parse_pmc_full_text_xml(xml_content: bytes, requested_pmcids: List[str]) -> Dict[str, str]:
    """
    从 EFetch 返回的 PMC 全文 XML 中解析正文内容。

    Args:
        xml_content (bytes): EFetch API 返回的 XML 响应内容。
        requested_pmcids (List[str]): 原始请求的 PMCID 列表，用于初始化结果字典。

    Returns:
        Dict[str, str]: 一个字典，键是 PMCID，值是提取的文章正文文本或错误/状态消息。
    """
    full_texts = {pmcid: "Article not found or error during fetch." for pmcid in requested_pmcids} # 初始化，假设都失败

    if not xml_content:
        logger.warning("Received empty content for PMC full text parsing.")
        return full_texts # 返回初始化的失败状态

    try:
        root = ET.fromstring(xml_content)
        # EFetch 对 PMC 的响应通常是一个 <pmc-articleset> 包含多个 <article>
        articles_found_in_response = {}
        for article in root.findall('.//article'):
            pmcid_in_response = None
            # 尝试从 <front>/<article-meta>/<article-id> 获取 PMCID
            article_meta = article.find('.//front/article-meta')
            if article_meta is not None:
                for article_id in article_meta.findall('.//article-id[@pub-id-type="pmc"]'):
                    pmcid_in_response = article_id.text
                    if pmcid_in_response and pmcid_in_response.startswith("PMC"):
                        # 确保格式正确，EFetch有时返回纯数字，需要加上"PMC"前缀
                        # 但如果已经是 PMCxxxx 了，就直接用
                        pass
                    elif pmcid_in_response and pmcid_in_response.isdigit():
                        pmcid_in_response = f"PMC{pmcid_in_response}"
                    else:
                        pmcid_in_response = None # 无效格式

                    if pmcid_in_response:
                        break # 找到 PMCID

            if not pmcid_in_response:
                logger.warning("Found an article in the PMC full text response without a parsable PMCID.")
                continue

            # 提取 <body> 内容
            body_text = f"Full text body not found or not available in XML for {pmcid_in_response}." # 默认消息
            body_element = article.find('.//body')
            if body_element is not None:
                # 基础文本提取：连接 <body> 内所有元素的文本内容
                # 注意：这会包含所有子标签的文本，可能需要更复杂的逻辑来处理格式
                text_parts = [text.strip() for text in body_element.itertext() if text.strip()]
                extracted_text = "\n\n".join(text_parts) # 使用双换行符分隔提取的文本块，模仿段落

                if extracted_text: # 确保提取到了内容
                    body_text = extracted_text
                else:
                    body_text = f"Found body element for {pmcid_in_response}, but it contained no extractable text."

            articles_found_in_response[pmcid_in_response] = body_text

        # 更新结果字典
        for pmcid in requested_pmcids:
            if pmcid in articles_found_in_response:
                full_texts[pmcid] = articles_found_in_response[pmcid]
            # else: 保持初始化的 "Article not found..." 消息

    except ET.ParseError as e:
        logger.error(f"Failed to parse PMC full text XML: {e}", exc_info=True)
        # 如果解析失败，所有请求的 ID 都标记为解析错误
        for pmcid in requested_pmcids:
             full_texts[pmcid] = f"Error parsing XML response: {e}"
    except Exception as e:
        logger.error(f"Error processing PMC full text XML data: {e}", exc_info=True)
        for pmcid in requested_pmcids:
             full_texts[pmcid] = f"Unexpected error during XML processing: {e}"

    return full_texts


# --- MCP 工具函数 ---

@mcp.tool()
async def search_pubmed(
    query: str,
    retmax: int = 10,
    sort: str = 'relevance',
    mindate: Optional[str] = None,
    maxdate: Optional[str] = None
) -> Dict[str, Any]:
    """
    在 PubMed 数据库中搜索文章摘要和元数据。

    Args:
        query (str): 检索关键词或语句。
        retmax (int): 返回的最大条目数 (默认 10)。
        sort (str): 排序方式 ('relevance' 或 'date') (默认 'relevance')。 'date' 表示按最新发布日期排序。
        mindate (str, optional): 最早发布日期 (格式: YYYY/MM/DD)。
        maxdate (str, optional): 最晚发布日期 (格式: YYYY/MM/DD)。

    Returns:
        Dict[str, Any]: 包含结果列表或错误信息的字典。
                        成功: {"success": True, "results": [{"pmid": ..., "title": ..., "abstract": ..., "journal": ..., "doi": ..., "pmcid": ...}]}
                        失败: {"success": False, "error": "错误信息"}
    """
    logger.info(f"Executing search_pubmed: query='{query}', retmax={retmax}, sort={sort}, mindate={mindate}, maxdate={maxdate}")
    try:
        # 1. ESearch: 获取 PMIDs
        search_url = f"{BASE_URL}/esearch.fcgi"
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': retmax,
            'sort': 'pub+date' if sort == 'date' else 'relevance',
            'retmode': 'json', # 使用 JSON 获取 ID 列表更简单
            'usehistory': 'y' # 理论上可以用 history 获取，但直接传 IDs 更简单
        }
        if mindate: search_params['mindate'] = mindate
        if maxdate: search_params['maxdate'] = maxdate
        if mindate or maxdate: search_params['datetype'] = 'pdat'

        search_response = make_request_with_retry(search_url, search_params)
        search_data = search_response.json()

        pmids = search_data.get("esearchresult", {}).get("idlist", [])

        if not pmids:
            logger.info("No PMIDs found.")
            return {"success": True, "results": []}

        logger.info(f"Found {len(pmids)} PMIDs.")

        # 2. EFetch: 获取详情 (摘要等元数据)
        fetch_url = f"{BASE_URL}/efetch.fcgi"
        # 对大量 IDs，最好用 POST
        fetch_params = {
            'db': 'pubmed',
            'id': ",".join(pmids), # GET/POST 都接受逗号分隔的 ID 列表
            'retmode': 'xml',
            'rettype': 'abstract' # 获取摘要信息
        }
        fetch_method = 'post' if len(pmids) > 200 else 'get' # 调整 POST 阈值

        fetch_response = make_request_with_retry(fetch_url, fetch_params, method=fetch_method)

        # 3. 解析 XML 结果 (现在包含 PMCID)
        articles = parse_pubmed_xml_details(fetch_response.content)

        logger.info(f"Successfully parsed {len(articles)} articles metadata.")
        return {"success": True, "results": articles}

    except requests.exceptions.RequestException as e:
         logger.error(f"HTTP request error during search_pubmed: {e}", exc_info=True)
         error_msg = str(e)
         if e.response is not None:
             try: error_detail = e.response.text; error_msg = f"{error_msg} - Response: {error_detail[:500]}"
             except: pass
         return {"success": False, "error": f"Network or API error: {error_msg}"}
    except ET.ParseError as e:
        logger.error(f"XML parsing error during search_pubmed: {e}", exc_info=True)
        return {"success": False, "error": f"Failed to parse NCBI XML response: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error in search_pubmed: {e}", exc_info=True)
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}


@mcp.tool()
async def search_pmc(
    query: str,
    retmax: int = 10,
    sort: str = 'relevance',
    mindate: Optional[str] = None,
    maxdate: Optional[str] = None
) -> Dict[str, Any]:
    """
    在 PubMed Central (PMC) 数据库中搜索文章并返回标题和链接。

    Args:
        query (str): 检索关键词或语句。
        retmax (int): 返回的最大条目数 (默认 10)。
        sort (str): 排序方式 ('relevance' 或 'date') (默认 'relevance')。 'date' 表示按最新发布日期排序。
        mindate (str, optional): 最早发布日期 (格式: YYYY/MM/DD)。
        maxdate (str, optional): 最晚发布日期 (格式: YYYY/MM/DD)。

    Returns:
        Dict[str, Any]: 包含结果列表或错误信息的字典。
                        成功: {"success": True, "results": [{"title": ..., "pmc_link": ...}]}
                        失败: {"success": False, "error": "错误信息"}
    """
    logger.info(f"Executing search_pmc: query='{query}', retmax={retmax}, sort={sort}, mindate={mindate}, maxdate={maxdate}")
    try:
        # 1. ESearch: 在 PMC 数据库搜索 UIDs
        search_url = f"{BASE_URL}/esearch.fcgi"
        search_params = {
            'db': 'pmc',
            'term': query,
            'retmax': retmax,
            'sort': 'pub date' if sort == 'date' else 'relevance', # PMC 文档确认用 'pub date'
            'retmode': 'json',
            'usehistory': 'y'
        }
        if mindate: search_params['mindate'] = mindate
        if maxdate: search_params['maxdate'] = maxdate
        if mindate or maxdate: search_params['datetype'] = 'pdat'

        search_response = make_request_with_retry(search_url, search_params)
        search_data = search_response.json()

        pmc_uids = search_data.get("esearchresult", {}).get("idlist", [])

        if not pmc_uids:
            logger.info("No PMC UIDs found.")
            return {"success": True, "results": []}

        logger.info(f"Found {len(pmc_uids)} potential PMC UIDs.")

        # 2. ESummary: 获取标题和 PMCID (用于构建链接)
        summary_url = f"{BASE_URL}/esummary.fcgi"
        summary_params = {
            'db': 'pmc',
            'id': ",".join(pmc_uids),
            'retmode': 'xml'
        }
        summary_method = 'post' if len(pmc_uids) > 200 else 'get'

        summary_response = make_request_with_retry(summary_url, summary_params, method=summary_method)

        # 3. 解析 ESummary XML 结果
        articles = parse_pmc_summary_xml(summary_response.content)

        logger.info(f"Successfully parsed {len(articles)} PMC articles with links.")
        return {"success": True, "results": articles}

    except requests.exceptions.RequestException as e:
         logger.error(f"HTTP request error during search_pmc: {e}", exc_info=True)
         error_msg = str(e)
         if e.response is not None:
             try: error_detail = e.response.text; error_msg = f"{error_msg} - Response: {error_detail[:500]}"
             except: pass
         return {"success": False, "error": f"Network or API error: {error_msg}"}
    except ET.ParseError as e:
        logger.error(f"XML parsing error during search_pmc: {e}", exc_info=True)
        return {"success": False, "error": f"Failed to parse NCBI XML response: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error in search_pmc: {e}", exc_info=True)
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}


# --- 新增：获取 PMC 全文的 MCP 工具 ---
@mcp.tool()
async def pmcid_to_full_article(pmcids: List[str]) -> Dict[str, Any]:
    """
    根据提供的 PMCID 列表，尝试从 PubMed Central (PMC) 获取文章的全文内容。

    注意：并非所有文章都有可用的 XML 全文。此函数仅返回可获取和解析的部分。

    Args:
        pmcids (List[str]): 需要获取全文的文章的 PMCID 列表 (例如 ["PMC1234567", "PMC7654321"])。
                           建议一次请求的 ID 数量不要过多 (例如少于 200 个)。

    Returns:
        Dict[str, Any]: 包含操作状态和结果的字典。
                        成功: {"success": True, "results": {"PMC1234567": "文章正文...", "PMC7654321": "Full text body not found..."}}
                              其中 results 是一个字典，键是输入的 PMCID，值是提取的全文或状态消息。
                        失败: {"success": False, "error": "错误信息"}
    """
    if not pmcids:
        logger.warning("pmcid_to_full_article called with empty list.")
        return {"success": True, "results": {}} # 没有输入，返回空结果

    # 验证 PMCID 格式 (可选但推荐)
    valid_pmcids = []
    invalid_formats = {}
    pmcid_pattern = re.compile(r'^PMC\d+(\.\d+)?$', re.IGNORECASE)
    for pmcid in pmcids:
        if isinstance(pmcid, str) and pmcid_pattern.match(pmcid):
             # EFetch 通常需要不带 "PMC" 前缀的数字 ID，但文档建议带上也没问题。
             # 为了保险，我们传递不带前缀的ID。
             # numeric_id = pmcid[3:] # 去掉 "PMC"
             # valid_pmcids.append(numeric_id)
             # 更新：文档说直接用 PMCID 也可以，甚至更好。我们直接用。
             valid_pmcids.append(pmcid)
        else:
            invalid_formats[pmcid] = "Invalid PMCID format."

    if not valid_pmcids:
         logger.error("No valid PMCIDs provided after format check.")
         return {"success": True, "results": invalid_formats} # 返回格式错误信息

    logger.info(f"Executing pmcid_to_full_article for {len(valid_pmcids)} valid PMCIDs: {valid_pmcids}")

    # 使用 EFetch 获取全文
    fetch_url = f"{BASE_URL}/efetch.fcgi"
    fetch_params = {
        'db': 'pmc',
        'id': ",".join(valid_pmcids),
        'retmode': 'xml',
        'rettype': 'full' # 尝试获取全文 XML
    }
    fetch_method = 'post' if len(valid_pmcids) > 50 else 'get' # 全文请求可能更大，降低 POST 阈值

    try:
        fetch_response = make_request_with_retry(fetch_url, fetch_params, method=fetch_method)

        # 解析返回的 XML 以获取全文
        # 需要传入原始请求的 PMCID (带 "PMC" 前缀) 以便结果映射
        full_texts = parse_pmc_full_text_xml(fetch_response.content, pmcids)

        # 合并格式错误的结果
        full_texts.update(invalid_formats)

        return {"success": True, "results": full_texts}

    except requests.exceptions.RequestException as e:
         logger.error(f"HTTP request error during pmcid_to_full_article: {e}", exc_info=True)
         error_msg = f"Network or API error: {e}"
         if e.response is not None:
             try: error_detail = e.response.text; error_msg = f"{error_msg} - Response: {error_detail[:500]}"
             except: pass
         # 将错误信息应用到所有请求的有效 ID 上
         error_results = {pmcid: error_msg for pmcid in valid_pmcids}
         error_results.update(invalid_formats)
         return {"success": False, "error": error_msg, "results": error_results} # 也可在 results 中反映错误
    except ET.ParseError as e:
        logger.error(f"XML parsing error during pmcid_to_full_article: {e}", exc_info=True)
        error_msg = f"Failed to parse NCBI XML response: {e}"
        error_results = {pmcid: error_msg for pmcid in valid_pmcids}
        error_results.update(invalid_formats)
        return {"success": False, "error": error_msg, "results": error_results}
    except Exception as e:
        logger.error(f"Unexpected error in pmcid_to_full_article: {e}", exc_info=True)
        error_msg = f"An unexpected error occurred: {str(e)}"
        error_results = {pmcid: error_msg for pmcid in valid_pmcids}
        error_results.update(invalid_formats)
        return {"success": False, "error": error_msg, "results": error_results}


# --- 主程序入口 ---
if __name__ == "__main__":
    # 可以在这里添加一些启动前的检查或配置加载
    logger.info("Starting PubMed Awesome Tools MCP Server...")
    # 运行 MCP 服务器
    mcp.run()
    logger.info("MCP Server stopped.")

