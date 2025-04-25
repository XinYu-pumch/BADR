#!/usr/bin/env python
# -*- coding: utf-8 -*-
import traceback
import asyncio
import io
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple, Union # 添加 Tuple

import chromadb
# import numpy as np # 暂时不需要 numpy
import PyPDF2
import requests
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
import httpx # 显式导入 httpx

# --- 配置 ---
# PubMed API
NCBI_EMAIL = "your.email@example.com" # 替换成你的邮箱
NCBI_API_KEY = None # 可选

# 硅基流动 Embedding API
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/embeddings"
# !!! 安全警告：切勿在生产代码中硬编码 API Key !!!
# 建议使用环境变量或其他安全方式管理
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "xxxxxxxxxxxx") #把xxx替换为你的硅基流动的api_key


SILICONFLOW_EMBEDDING_MODEL = "Pro/BAAI/bge-m3"

MAX_CONCURRENT_EMBEDDING_REQUESTS = 3 # 并发 Embedding 请求限制

# Embedding 模型配置
CONTENT_CHUNK_SIZE = 2048      # 文献内容分块的目标字符数
CONTENT_CHUNK_OVERLAP = 128    # 文献内容分块之间的重叠字符数
MAX_CHARS_FOR_EMBEDDING_API_LIMIT = 15000 # 保守估计的模型单次输入字符上限 (用于内部检查)

# ChromaDB
CHROMA_DB_PATH = "xxxxxxxxxx" # xxxx为你的chromadb的临时路径
DEFAULT_RETMAX_PUBMED = 20 # 默认检索数量
PDF_PARSE_TIMEOUT = 60.0 # PDF 解析超时时间（秒）
ALL_ABSTRACTS_DOC_ID = "doc_all_abstracts" # 用于存储所有摘要整合文档的固定 ID
COMBINED_ABSTRACT_METADATA_KEY = "full_abstract_text" # [新增] 存储整合摘要的 metadata key

# Sci-Hub
SCIHUB_MIRRORS = [
    "https://sci-hub.se",
    "https://sci-hub.st",
    "https://sci-hub.ru",
]

# --- 日志配置 ---
# (日志配置部分保持不变)
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, "literature_search_mcp_server.log")
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    try:
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging initialized. Log file: {log_file_path}")
    except Exception as e:
        logger.error(f"Failed to initialize file logger at {log_file_path}: {e}", exc_info=True)
else:
    logger.info("Logger already initialized.")

# MCP 服务器实例
mcp = FastMCP("LiteratureSearchTool")

# --- PubMed 辅助函数 ---
# (make_request_with_retry, parse_pubmed_xml_details, parse_pmc_full_text_xml, _search_pubmed_internal, _get_pmc_full_text_internal 保持不变)
def make_request_with_retry(url: str, params: Dict[str, Any], method: str = 'get', max_retries: int = 3, wait_time: float = 1.0) -> requests.Response:
    """带重试的 HTTP 请求"""
    effective_params = params.copy()
    if NCBI_EMAIL and NCBI_EMAIL != "your.email@example.com":
        effective_params['email'] = NCBI_EMAIL
    if NCBI_API_KEY:
        effective_params['api_key'] = NCBI_API_KEY

    response = None
    for attempt in range(max_retries):
        try:
            if method.lower() == 'get':
                response = requests.get(url, params=effective_params, timeout=30)
            elif method.lower() == 'post':
                 response = requests.post(url, data=effective_params, timeout=60)
            else: raise ValueError(f"Unsupported HTTP method: {method}")
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout as e:
             logger.warning(f"Request timed out (attempt {attempt + 1}/{max_retries}): {str(e)}")
             if attempt == max_retries - 1: raise
             time.sleep(wait_time); wait_time *= 2
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if response is not None and 400 <= response.status_code < 500: raise # 客户端错误不重试
            if attempt == max_retries - 1: raise
            time.sleep(wait_time); wait_time *= 2
    raise Exception("Request failed after multiple retries")

def parse_pubmed_xml_details(xml_content: bytes) -> List[Dict[str, Any]]:
    """解析 PubMed EFetch XML 获取元数据 (含 DOI, PMCID, 完整摘要)"""
    results = []
    if not xml_content: return results
    try:
        root = ET.fromstring(xml_content)
        for article in root.findall('.//PubmedArticle'):
            entry = {'pmid': None, 'title': 'N/A', 'abstract': None, 'doi': None, 'pmcid': None}
            pmid_elem = article.find('.//PMID')
            if pmid_elem is not None: entry['pmid'] = pmid_elem.text

            title_elem = article.find('.//ArticleTitle')
            if title_elem is not None: entry['title'] = "".join(title_elem.itertext()).strip()

            # 获取完整摘要
            abstract_texts = []
            abstract_node = article.find('.//Abstract')
            if abstract_node is not None:
                 for abst_text in abstract_node.findall('./AbstractText'):
                     label = abst_text.get('Label')
                     text = "".join(t for t in abst_text.itertext()).strip()
                     if text: abstract_texts.append(f"{label}: {text}" if label else text)
            if abstract_texts: entry['abstract'] = "\n".join(abstract_texts) # 存储完整摘要

            doi, pmcid = None, None
            article_id_list_node = article.find('.//ArticleIdList')
            if article_id_list_node is not None:
                for item in article_id_list_node.findall('./ArticleId'):
                    id_type = item.get('IdType')
                    if id_type == 'doi': doi = item.text
                    elif id_type == 'pmc': pmcid = item.text
            if not doi:
                 doi_elem = article.find(".//ELocationID[@EIdType='doi']")
                 if doi_elem is not None: doi = doi_elem.text

            entry['doi'] = doi
            entry['pmcid'] = pmcid
            if entry['pmid']: results.append(entry)
    except ET.ParseError as e: logger.error(f"Failed to parse PubMed XML: {e}", exc_info=True); raise
    except Exception as e: logger.error(f"Error processing PubMed XML: {e}", exc_info=True); raise
    return results

def parse_pmc_full_text_xml(xml_content: bytes, requested_pmcids: List[str]) -> Dict[str, str]:
    """解析 PMC 全文 XML"""
    full_texts = {pmcid: f"Error fetching or parsing full text for {pmcid}." for pmcid in requested_pmcids}
    if not xml_content: return full_texts
    try:
        root = ET.fromstring(xml_content)
        articles_found_in_response = {}
        for article in root.findall('.//article'):
            pmcid_in_response = None
            article_meta = article.find('.//front/article-meta')
            if article_meta is not None:
                for article_id in article_meta.findall('.//article-id[@pub-id-type="pmc"]'):
                    raw_pmcid = article_id.text
                    if raw_pmcid and raw_pmcid.isdigit(): pmcid_in_response = f"PMC{raw_pmcid}"
                    elif raw_pmcid and raw_pmcid.startswith("PMC"): pmcid_in_response = raw_pmcid
                    if pmcid_in_response: break
            if not pmcid_in_response: continue

            body_text = f"Full text body not found or empty in XML for {pmcid_in_response}."
            body_element = article.find('.//body')
            if body_element is not None:
                text_parts = []
                for element in body_element.iter():
                    if element.text:
                        text_parts.append(element.text.strip())
                    if element.tag in ['p', 'sec', 'title', 'break', 'list-item']:
                        text_parts.append("\n") # 添加换行标记
                extracted_text = ' '.join(text_parts).strip() # 先用空格连接
                extracted_text = re.sub(r'\s*\n\s*', '\n', extracted_text) # 合并多余换行
                extracted_text = re.sub(r'[ \t]+', ' ', extracted_text) # 合并多余空格
                if extracted_text: body_text = extracted_text

            articles_found_in_response[pmcid_in_response] = body_text

        for pmcid in requested_pmcids:
            if pmcid in articles_found_in_response: full_texts[pmcid] = articles_found_in_response[pmcid]
            else: full_texts[pmcid] = f"Article {pmcid} not found in the response."

    except ET.ParseError as e:
        logger.error(f"Failed to parse PMC full text XML: {e}", exc_info=True)
        for pmcid in requested_pmcids: full_texts[pmcid] = f"Error parsing XML response: {e}"
    except Exception as e:
        logger.error(f"Error processing PMC full text XML data: {e}", exc_info=True)
        for pmcid in requested_pmcids: full_texts[pmcid] = f"Unexpected error during XML processing: {e}"
    return full_texts

async def _search_pubmed_internal(
    query: str,
    retmax: int,
    sort_by: str,
    min_date: Optional[str],
    max_date: Optional[str]
) -> List[Dict[str, Any]]:
    """
    内部函数：搜索 PubMed，支持排序和日期过滤。
    """
    logger.info(f"Searching PubMed: query='{query}', retmax={retmax}, sort='{sort_by}', min_date='{min_date}', max_date='{max_date}'")
    articles = []
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {'db': 'pubmed', 'term': query, 'retmax': retmax, 'retmode': 'json'}
        if sort_by == "pub_date": search_params['sort'] = 'pub date'
        elif sort_by == "relevance": search_params['sort'] = 'relevance'
        else: logger.warning(f"Invalid sort_by value '{sort_by}'. Defaulting to 'relevance'."); search_params['sort'] = 'relevance'
        if min_date or max_date:
            search_params['datetype'] = 'pdat'
            if min_date: search_params['mindate'] = min_date
            if max_date: search_params['maxdate'] = max_date

        search_response = await asyncio.to_thread(make_request_with_retry, search_url, search_params)
        search_data = search_response.json()
        pmids = search_data.get("esearchresult", {}).get("idlist", [])

        if not pmids: logger.info("No PMIDs found."); return []
        logger.info(f"Found {len(pmids)} PMIDs.")

        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {'db': 'pubmed', 'id': ",".join(pmids), 'retmode': 'xml', 'rettype': 'abstract'}
        fetch_method = 'post' if len(pmids) > 200 else 'get'
        fetch_response = await asyncio.to_thread(make_request_with_retry, fetch_url, fetch_params, method=fetch_method)

        articles = await asyncio.to_thread(parse_pubmed_xml_details, fetch_response.content)
        logger.info(f"Successfully parsed metadata for {len(articles)} articles.")
    except Exception as e: logger.error(f"Error during PubMed search: {e}", exc_info=True)
    return articles

async def _get_pmc_full_text_internal(pmcids: List[str]) -> Dict[str, str]:
    """内部函数：获取 PMC 全文"""
    if not pmcids: return {}
    logger.info(f"Fetching PMC full text for {len(pmcids)} PMCID(s): {pmcids}")
    full_texts = {pmcid: f"Error fetching full text for {pmcid}." for pmcid in pmcids}
    try:
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {'db': 'pmc', 'id': ",".join(pmcids), 'retmode': 'xml', 'rettype': 'full'}
        fetch_method = 'post' if len(pmcids) > 50 else 'get'
        fetch_response = await asyncio.to_thread(make_request_with_retry, fetch_url, fetch_params, method=fetch_method)
        parsed_texts = await asyncio.to_thread(parse_pmc_full_text_xml, fetch_response.content, pmcids)
        return parsed_texts
    except Exception as e:
        logger.error(f"Error fetching PMC full text: {e}", exc_info=True)
        error_msg = f"Error fetching full text: {e}"
        for pmcid in pmcids: full_texts[pmcid] = error_msg
    return full_texts


# --- Sci-Hub & PDF 辅助函数 ---
# (run_scihub_web_request, extract_text_from_pdf_stream, _get_pdf_text_from_url_internal 函数保持不变)
def run_scihub_web_request(doi: str) -> Optional[str]:
    """尝试从 Sci-Hub 镜像获取 PDF 链接"""
    for mirror in SCIHUB_MIRRORS:
        try:
            scihub_url = f"{mirror}/{doi}"
            logger.info(f"Trying Sci-Hub mirror: {scihub_url}")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(scihub_url, timeout=15, headers=headers, allow_redirects=True) # 允许重定向
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'lxml')
                pdf_url = None
                embed = soup.find('embed', type='application/pdf')
                if embed and embed.get('src'): pdf_url = embed['src']
                if not pdf_url:
                    iframe = soup.find('iframe', id='pdf') # 通用 ID
                    if iframe and iframe.get('src'): pdf_url = iframe.get('src')
                    else:
                        iframes = soup.find_all('iframe')
                        for frame in iframes:
                            src = frame.get('src', '')
                            if src and 'pdf' in src.lower():
                                pdf_url = src; break
                if not pdf_url:
                    button = soup.find('button', onclick=lambda x: x and 'location.href=' in x)
                    if button:
                        match = re.search(r"location\.href='([^']+)'", button['onclick'])
                        if match: pdf_url = match.group(1)
                if not pdf_url:
                    links = soup.find_all('a', href=True)
                    for link in links:
                         href = link['href']
                         if href and href.lower().endswith('.pdf') and ('download' in href.lower() or 'pdf' in href.lower()):
                             pdf_url = href; break
                         elif href and 'pdf' in href.lower() and (href.startswith('http') or href.startswith('//')):
                             pdf_url = href; break

                if pdf_url:
                    if pdf_url.startswith('//'): pdf_url = 'https:' + pdf_url
                    elif pdf_url.startswith('/'):
                        base_url = '/'.join(response.url.split('/')[:3])
                        pdf_url = base_url + pdf_url
                    elif not pdf_url.startswith('http'):
                         base_url = '/'.join(response.url.split('/')[:-1])
                         pdf_url = base_url + '/' + pdf_url
                    logger.info(f"Found PDF link via Sci-Hub: {pdf_url}")
                    return pdf_url
                elif "article not found" in response.text.lower():
                     logger.info(f"Article not found on Sci-Hub mirror {mirror} for DOI {doi}")
                     return None #明确返回None，而不是让循环继续
                else: logger.warning(f"Could not find PDF link structure on page {scihub_url}")
            else: logger.warning(f"Sci-Hub mirror {mirror} failed for DOI {doi}, Status: {response.status_code}")
        except requests.exceptions.Timeout:
             logger.warning(f"Timeout accessing Sci-Hub mirror {mirror} for DOI {doi}")
        except Exception as e: logger.error(f"Error processing Sci-Hub for DOI {doi} from {mirror}: {str(e)}")
    logger.warning(f"Failed to get PDF link from all Sci-Hub mirrors for DOI: {doi}")
    return None

def extract_text_from_pdf_stream(pdf_stream: io.BytesIO, url: str) -> Optional[str]:
    """从 PDF 文件流中提取文本"""
    try:
        reader = PyPDF2.PdfReader(pdf_stream, strict=False) # 使用 strict=False 增加容错性
        if reader.is_encrypted:
            logger.warning(f"PDF {url} is encrypted. Attempting to decrypt with default password.")
            try: reader.decrypt('')
            except Exception as decrypt_err:
                 logger.error(f"Failed to decrypt PDF {url}: {decrypt_err}")
                 return None

        num_pages = len(reader.pages)
        logger.debug(f"PDF {url} has {num_pages} pages.")
        text_parts = []
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text: text_parts.append(page_text.strip())
            except Exception as page_err:
                logger.warning(f"Error extracting text from page {i+1} of PDF {url}: {page_err}")
                continue

        full_text = "\n\n".join(text_parts)
        # 简单的文本清理
        full_text = re.sub(r'\n{3,}', '\n\n', full_text) # 最多保留两个换行符
        full_text = re.sub(r'[ \t]{2,}', ' ', full_text) # 合并多个空格/制表符为一个空格
        full_text = full_text.strip()

        if full_text:
             logger.info(f"Successfully extracted text from PDF: {url}")
             return full_text
        else:
             logger.warning(f"Extracted empty text from PDF: {url}")
             return None

    except PyPDF2.errors.PdfReadError as pdf_err:
        logger.error(f"PyPDF2 error reading PDF {url}: {pdf_err}")
        return None
    except Exception as e:
         logger.error(f"Unexpected error extracting text from PDF {url}: {str(e)}", exc_info=True)
         return None

async def _get_pdf_text_from_url_internal(pdf_url: str, parse_timeout: float = PDF_PARSE_TIMEOUT) -> Optional[str]:
    """内部函数：下载 PDF 并提取文本，增加解析超时"""
    if not pdf_url: return None
    logger.info(f"Attempting to read PDF from URL: {pdf_url}")
    extracted_text = None
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        # 使用 httpx 进行异步下载
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(pdf_url, headers=headers)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' not in content_type and 'octet-stream' not in content_type:
                logger.warning(f"URL {pdf_url} Content-Type ('{content_type}') may not be PDF, proceeding anyway.")
            pdf_content = await response.aread() # 异步读取内容

        logger.debug(f"PDF content size for {pdf_url}: {len(pdf_content)} bytes")

        if not isinstance(pdf_content, bytes):
             logger.error(f"Expected bytes from response content for {pdf_url}, but got {type(pdf_content)}. Skipping.")
             return None
        if not pdf_content:
             logger.warning(f"PDF content is empty for {pdf_url}. Skipping.")
             return None

        pdf_stream = io.BytesIO(pdf_content)
        logger.info(f"Starting PDF text extraction for {pdf_url} (timeout: {parse_timeout}s)")
        try:
            extracted_text = await asyncio.wait_for(
                asyncio.to_thread(extract_text_from_pdf_stream, pdf_stream, pdf_url),
                timeout=parse_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"PDF parsing timed out after {parse_timeout} seconds for {pdf_url}")
            return None
        except Exception as parse_exc:
             logger.error(f"Error during PDF text extraction thread for {pdf_url}: {parse_exc}", exc_info=True)
             return None

        return extracted_text

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
             logger.warning(f"PDF not found (404) at {pdf_url}")
        else: logger.error(f"HTTP error downloading PDF from {pdf_url}: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.RequestError as e:
         logger.error(f"Failed to download PDF from {pdf_url}: {type(e).__name__} - {str(e)}")
         return None
    except Exception as e:
         logger.error(f"Unexpected error processing PDF (download/prep) from {pdf_url}: {str(e)}", exc_info=True)
         return None


# --- Embedding 辅助函数 ---

def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """将长文本按目标字符数分割成带重叠的块"""
    # (此函数保持不变)
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start_index = 0
    text_len = len(text)
    while start_index < text_len:
        end_index = start_index + chunk_size
        chunk = text[start_index:end_index]
        chunks.append(chunk)
        start_index += (chunk_size - overlap)
        if start_index >= text_len: break
        if start_index <= (len(chunks) - 1) * (chunk_size - overlap):
            logger.warning(f"Potential overlap issue detected in chunking. Forcing start_index forward.")
            start_index = (len(chunks) - 1) * (chunk_size - overlap) + 1

    return [c for c in chunks if c and c.strip()]


async def _get_single_embedding(
    text: str,
    api_key: str,
    api_url: str,
    model: str,
    # [新增] 显式传递 client，避免重复创建
    client: httpx.AsyncClient
) -> Optional[List[float]]:
    """[内部] 使用 API 获取单段文本的嵌入 (需要传入 httpx client)"""
    if not text or not text.strip():
        logger.warning("Skipping embedding for empty or whitespace-only text.")
        return None

    if len(text) > MAX_CHARS_FOR_EMBEDDING_API_LIMIT:
        logger.error(f"Chunk text length ({len(text)}) exceeds API limit ({MAX_CHARS_FOR_EMBEDDING_API_LIMIT}). Truncating. Text: {text[:100]}...")
        text = text[:MAX_CHARS_FOR_EMBEDDING_API_LIMIT] # 进行截断以尝试获取部分结果

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"input": [text], "model": model}
    logger.debug(f"Requesting single embedding for text chunk (length {len(text)}): {text[:50]}...")

    try:
        # [修改] 使用传入的 client
        response = await client.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()

        if "data" in response_data and response_data["data"] and "embedding" in response_data["data"][0]:
             embedding_vector = response_data["data"][0]["embedding"]
             if isinstance(embedding_vector, list) and all(isinstance(n, (int, float)) for n in embedding_vector):
                 logger.debug(f"Successfully obtained single embedding for chunk, vector dimension: {len(embedding_vector)}")
                 return embedding_vector
             else:
                 logger.error(f"Embedding API returned invalid vector type for chunk {text[:50]}...: {type(embedding_vector)}")
                 return None
        else:
            logger.error(f"Unexpected Embedding API response structure for chunk: {response_data}")
            return None
    except httpx.RequestError as req_err:
         logger.error(f"httpx request error getting embedding for chunk: {req_err}", exc_info=True)
         return None
    except httpx.HTTPStatusError as status_err:
         logger.error(f"HTTP error getting single embedding for chunk: {status_err.response.status_code} - {status_err.response.text}", exc_info=True)
         return None
    except Exception as e: logger.error(f"Unexpected error getting single embedding for chunk: {e}", exc_info=True); return None


# [新增] 动态获取 Embedding 维度的辅助函数
async def get_embedding_dimension(
    api_key: str,
    api_url: str,
    model_name: str,
    client: httpx.AsyncClient # [新增] 接收 client
) -> Optional[int]:
    """通过嵌入一个测试字符串来动态获取模型的输出维度。"""
    test_string = "dimension check"
    logger.info(f"Performing a test embedding with model '{model_name}' to determine dimension...")
    try:
        # [修改] 调用 _get_single_embedding 并传入 client
        test_embedding = await _get_single_embedding(test_string, api_key, api_url, model_name, client)

        if test_embedding and isinstance(test_embedding, list) and len(test_embedding) > 0:
            dimension = len(test_embedding)
            logger.info(f"Successfully determined embedding dimension: {dimension}")
            return dimension
        else:
            logger.error("Failed to get a valid embedding during dimension check. Cannot determine dimension.")
            return None
    except Exception as e:
        logger.error(f"Error during embedding dimension check: {e}", exc_info=True)
        return None


# --- 主 MCP 工具函数 ---

@mcp.tool()
async def search_literature(
    keyword: str,
    db_name: str,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    sort_by: str = "relevance",
    num_results: int = DEFAULT_RETMAX_PUBMED
) -> Dict[str, Any]:
    """
    搜索文献、获取原文、将原文分块嵌入并存储到向量数据库。
    同时，将所有摘要合并存储为一个单独条目（使用占位符 Embedding 和 Metadata）。
    Embedding 维度将根据模型动态确定。

    Args:
        keyword (str): PubMed 搜索关键词。
        db_name (str): ChromaDB 集合名称。
        min_date (Optional[str]): 检索的起始出版日期 (格式: YYYY/MM/DD 或 YYYY)。默认为 None。
        max_date (Optional[str]): 检索的结束出版日期 (格式: YYYY/MM/DD 或 YYYY)。默认为 None。
        sort_by (str): 结果排序方式。可选: "relevance", "pub_date"。默认为 "relevance"。
        num_results (int): 希望检索的最大文献数量。默认为 DEFAULT_RETMAX_PUBMED。

    Returns:
        Dict[str, Any]: 操作结果摘要。
            {
                "success": bool,
                "message": str,
                "papers_found": int,        # PubMed 找到的文献数
                "papers_processed": int,    # 尝试处理的文献数 (获取内容+嵌入)
                "chunks_embedded": int,     # 成功嵌入并存储的文献块数量
                "abstracts_entry_saved": bool, # 是否成功保存了摘要合集条目
                "embedding_dimension": Optional[int], # 确定的或期望的维度
                "db_path": str,
                "collection_name": str,
                "errors": List[str]
            }
    """
    start_time = time.time()
    logger.info(f"Starting literature search: keyword='{keyword}', db_name='{db_name}', num_results={num_results}, sort='{sort_by}', min_date='{min_date}', max_date='{max_date}'")
    results = {
        "success": False, "message": "", "papers_found": 0, "papers_processed": 0,
        "chunks_embedded": 0, "abstracts_entry_saved": False,
        "embedding_dimension": None, # [新增] 记录维度
        "db_path": CHROMA_DB_PATH, "collection_name": db_name, "errors": []
    }
    # [新增] 创建一个贯穿请求的 httpx 客户端，用于所有 Embedding 调用
    async with httpx.AsyncClient(timeout=120.0) as http_client:
        try:
            # --- 1. 动态确定 Embedding 维度 ---
            determined_embedding_dimension = await get_embedding_dimension(
                SILICONFLOW_API_KEY, SILICONFLOW_API_URL, SILICONFLOW_EMBEDDING_MODEL, http_client
            )
            if determined_embedding_dimension is None:
                error_msg = f"Failed to determine embedding dimension for model '{SILICONFLOW_EMBEDDING_MODEL}'. Cannot proceed."
                logger.error(error_msg)
                results["message"] = error_msg; results["errors"].append(error_msg)
                return results # 无法确定维度，直接返回错误
            else:
                results["embedding_dimension"] = determined_embedding_dimension
                logger.info(f"Using dynamically determined embedding dimension: {determined_embedding_dimension}")


            # --- 2. 输入参数校验 ---
            if sort_by not in ["relevance", "pub_date"]:
                msg = f"Invalid sort_by value '{sort_by}'. Using default 'relevance'."
                logger.warning(msg); sort_by = "relevance"
            if num_results <= 0:
                msg = f"Invalid num_results value {num_results}. Using default {DEFAULT_RETMAX_PUBMED}."
                logger.warning(msg); num_results = DEFAULT_RETMAX_PUBMED

            # --- 3. 搜索 PubMed & 收集摘要 ---
            pubmed_results = await _search_pubmed_internal(
                keyword, retmax=num_results, sort_by=sort_by, min_date=min_date, max_date=max_date
            )
            results["papers_found"] = len(pubmed_results)
            if not pubmed_results:
                results["success"] = True; results["message"] = "PubMed search returned no results."
                logger.info(results["message"])
                return results

            all_abstracts_list = []
            for paper in pubmed_results:
                abstract = paper.get('abstract')
                if abstract and isinstance(abstract, str) and abstract.strip():
                    all_abstracts_list.append(abstract.strip())
            logger.info(f"Collected {len(all_abstracts_list)} non-empty abstracts from {results['papers_found']} papers.")

            # --- 4. 准备 ChromaDB (不再尝试在 metadata 中设置维度) ---
            try:
                os.makedirs(CHROMA_DB_PATH, exist_ok=True)
                chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

                # [修改] 移除 metadata 中的 "hnsw:dimension"
                logger.info(f"Attempting to get or create collection '{db_name}' with hnsw:space='cosine'. Dimension will be inferred.")
                collection = await asyncio.to_thread(
                    chroma_client.get_or_create_collection,
                    name=db_name,
                    metadata={"hnsw:space": "cosine"} # 只保留合法的 HNSW 空间参数
                    # ChromaDB 会从后续 upsert 的 embeddings 推断维度
                )
                logger.info(f"Successfully got or created ChromaDB collection '{db_name}' at '{CHROMA_DB_PATH}'. Expecting dimension {determined_embedding_dimension} to be inferred.")

                # [修改] 调整维度验证逻辑 - 可能在首次 upsert 前无法获取到维度
                # 我们可以尝试获取，但如果获取不到也不应视为致命错误，因为维度会在 upsert 时设置
                try:
                     collection_metadata = await asyncio.to_thread(getattr, collection, 'metadata') # 尝试获取 metadata
                     if collection_metadata:
                         current_dim = collection_metadata.get("hnsw:dimension") # 尝试获取维度
                         if current_dim: # 如果 metadata 中 *已经* 有维度信息 (例如集合已存在)
                             if current_dim != determined_embedding_dimension:
                                 # 维度不匹配仍然是严重问题
                                 error_msg = f"FATAL: Existing collection '{db_name}' dimension ({current_dim}) does not match dynamically determined dimension ({determined_embedding_dimension}). Delete the collection or fix configuration."
                                 logger.error(error_msg)
                                 results["message"] = error_msg; results["errors"].append(error_msg)
                                 return results # 必须停止
                             else:
                                 logger.info(f"Existing collection dimension ({current_dim}) matches determined dimension.")
                         else:
                              logger.warning(f"Collection '{db_name}' metadata exists but does not contain 'hnsw:dimension'. Dimension will be set upon first upsert.")
                     else:
                         logger.warning(f"Could not retrieve metadata for collection '{db_name}' immediately after creation/get. Dimension check skipped.")
                except Exception as meta_check_err:
                     logger.warning(f"Error checking collection metadata after get/create: {meta_check_err}. Dimension check skipped.")


            except ValueError as ve: # 明确捕获可能的 ValueError
                # 检查是否是由于其他原因导致的 ValueError
                if "Unknown HNSW parameter" in str(ve):
                     # 这个错误理论上不应该再发生，但以防万一
                     error_msg = f"Unexpected 'Unknown HNSW parameter' error even after removing hnsw:dimension: {ve}"
                else:
                     error_msg = f"ValueError during ChromaDB initialization for '{db_name}': {ve}"
                logger.error(error_msg, exc_info=True)
                results["message"] = error_msg; results["errors"].append(f"{error_msg} (Traceback logged)")
                return results
            except Exception as e: # 捕获其他可能的异常
                error_msg = f"Failed to initialize ChromaDB collection '{db_name}': {e}"
                logger.error(error_msg, exc_info=True)
                results["message"] = error_msg; results["errors"].append(f"{error_msg} (Traceback logged)")
                return results






            # --- 5. 存储合并摘要 (使用占位符 Embedding 和 Metadata) ---
            if all_abstracts_list:
                combined_abstract_string = "\n\n---\n\n".join(all_abstracts_list)
                combined_abstract_id = ALL_ABSTRACTS_DOC_ID
                placeholder_text = "Placeholder text for combined abstracts entry"

                try:
                    logger.info(f"Generating placeholder embedding for combined abstracts entry (ID: {combined_abstract_id}).")
                    # [修改] 调用 _get_single_embedding 并传入 client
                    placeholder_embedding = await _get_single_embedding(
                        placeholder_text, SILICONFLOW_API_KEY, SILICONFLOW_API_URL, SILICONFLOW_EMBEDDING_MODEL, http_client
                    )

                    if placeholder_embedding and len(placeholder_embedding) == determined_embedding_dimension:
                        # [修改] 将真实文本存储在 metadata 中
                        combined_abstract_metadata = {COMBINED_ABSTRACT_METADATA_KEY: combined_abstract_string}

                        logger.info(f"Attempting to save/update combined abstracts entry (ID: {combined_abstract_id}) using placeholder embedding and metadata.")
                        await asyncio.to_thread(
                            collection.upsert,
                            ids=[combined_abstract_id],
                            embeddings=[placeholder_embedding], # 使用占位符 embedding
                            metadatas=[combined_abstract_metadata], # 存储真实文本
                            documents=[placeholder_text] # 存储占位符文本
                        )
                        results["abstracts_entry_saved"] = True
                        logger.info(f"Successfully saved/updated the combined abstracts entry (ID: {combined_abstract_id}) with text stored in metadata.")
                    elif placeholder_embedding:
                         # 维度不匹配错误
                         error_msg = f"Placeholder embedding dimension ({len(placeholder_embedding)}) does not match determined dimension ({determined_embedding_dimension}). Cannot save combined abstracts."
                         logger.error(error_msg)
                         results["errors"].append(error_msg)
                         results["abstracts_entry_saved"] = False
                    else:
                        # 生成占位符 embedding 失败
                        error_msg = f"Failed to generate placeholder embedding for combined abstracts entry (ID: {combined_abstract_id}). Cannot save."
                        logger.error(error_msg)
                        results["errors"].append(error_msg)
                        results["abstracts_entry_saved"] = False

                except Exception as e:
                    error_msg = f"Failed to save combined abstracts entry (ID: {combined_abstract_id}): {e}"
                    logger.error(error_msg, exc_info=True)
                    results["errors"].append(f"{error_msg} (Traceback logged)")
                    results["abstracts_entry_saved"] = False
            else:
                logger.warning("No abstracts found to combine and save.")
                # 可选：删除旧条目
                try:
                     existing = await asyncio.to_thread(collection.get, ids=[ALL_ABSTRACTS_DOC_ID], limit=1, include=[])
                     if existing and existing['ids']:
                         await asyncio.to_thread(collection.delete, ids=[ALL_ABSTRACTS_DOC_ID])
                         logger.info(f"Deleted previous combined abstracts entry '{ALL_ABSTRACTS_DOC_ID}' as no new abstracts were found.")
                except Exception as e_del:
                     logger.warning(f"Could not check/delete previous combined abstracts entry '{ALL_ABSTRACTS_DOC_ID}': {e_del}")


            # --- 6. 获取 PMC 全文 ---
            pmcid_map = {p['pmcid']: p for p in pubmed_results if p.get('pmcid')}
            pmcids_to_fetch = list(pmcid_map.keys())
            pmc_full_texts = {}
            if pmcids_to_fetch:
                batch_size = 50
                for i in range(0, len(pmcids_to_fetch), batch_size):
                    batch_pmcids = pmcids_to_fetch[i:i+batch_size]
                    try:
                        batch_texts = await _get_pmc_full_text_internal(batch_pmcids)
                        pmc_full_texts.update(batch_texts)
                        logger.info(f"Fetched PMC texts for batch starting at index {i}")
                    except Exception as pmc_batch_err:
                        logger.error(f"Error fetching PMC text batch starting at index {i}: {pmc_batch_err}", exc_info=True)
                        error_detail = f"Error in batch fetch: {pmc_batch_err} (Traceback logged)"
                        results["errors"].append(f"PMC fetch failed for batch starting at {i}: {error_detail}")
                        for pmcid in batch_pmcids: pmc_full_texts[pmcid] = error_detail
                    await asyncio.sleep(0.5)

            # --- 7. 准备并执行文献内容处理任务 (获取内容 -> 分块 -> 嵌入) ---
            papers_to_process_input = []
            processed_paper_count = 0
            for paper in pubmed_results:
                processed_paper_count += 1
                paper_id = str(paper.get('pmid') or paper.get('doi'))
                if not paper_id or paper_id == 'None':
                    paper_id = f"temp_id_{time.time_ns()}_{processed_paper_count}"
                    logger.warning(f"Paper missing PMID and DOI, using temporary ID: {paper_id}. Title: {paper.get('title')}")
                else: paper_id = str(paper_id)

                text_content_from_pmc = None
                source = "N/A"
                pmcid = paper.get('pmcid')

                if pmcid and pmcid in pmc_full_texts:
                    full_text = pmc_full_texts[pmcid]
                    if full_text and isinstance(full_text, str) and not full_text.startswith(("Error", "Article not found", "Full text body not found")):
                        text_content_from_pmc = full_text; source = "PMC Full Text"
                        logger.info(f"Using PMC text for {paper_id} (PMCID: {pmcid})")
                    elif isinstance(full_text, str) and (full_text.startswith("Error") or full_text == f"Article {pmcid} not found in the response."):
                        error_detail = full_text
                        if "Error in batch fetch" not in error_detail: results["errors"].append(f"PMC content issue for {paper_id} (PMCID: {pmcid}): {error_detail}")
                        logger.warning(f"PMC text not available or error recorded for {paper_id} (PMCID: {pmcid}). Detail: {error_detail}")
                    else:
                        error_detail = f"PMC text unavailable or invalid ({type(full_text).__name__})" if full_text else "PMC text unavailable or invalid"
                        results["errors"].append(f"PMC fetch failed for {paper_id} (PMCID: {pmcid}): {error_detail}")
                        logger.warning(f"PMC text not available for {paper_id} (PMCID: {pmcid}). Detail: {error_detail}")

                paper_processing_data = {
                    "original_paper_id": paper_id,
                    "doi": paper.get('doi'),
                    "title": paper.get('title', 'N/A'),
                    "abstract": paper.get('abstract'),
                    "pmc_text": text_content_from_pmc,
                    "initial_source": source,
                    "http_client": http_client # [新增] 传递共享的 client
                }
                papers_to_process_input.append(paper_processing_data)

            tasks = []
            for paper_input in papers_to_process_input:
                tasks.append(asyncio.create_task(
                    fetch_content_and_process_chunks(paper_input) # 调用处理函数
                ))

            all_chunks_to_add = []
            processed_papers_successfully = 0

            try:
                logger.info(f"Gathering results for {len(tasks)} paper processing tasks...")
                raw_results = await asyncio.gather(*tasks, return_exceptions=True)
                logger.info("Finished gathering processing tasks.")

                results["papers_processed"] = len(raw_results)

                for i, result_or_exc in enumerate(raw_results):
                    paper_id_for_log = papers_to_process_input[i].get("original_paper_id", f"task_{i}")
                    if isinstance(result_or_exc, Exception):
                        error_msg = f"Error processing paper {paper_id_for_log}: {type(result_or_exc).__name__} - {result_or_exc}"
                        logger.error(f"Exception in gathered task for paper {paper_id_for_log}:", exc_info=result_or_exc)
                        results["errors"].append(f"{error_msg} (Traceback logged)")
                    elif result_or_exc is None:
                        logger.warning(f"Processing task for paper {paper_id_for_log} returned None (likely skipped or failed content/embedding).")
                        results["errors"].append(f"Processing skipped or failed (no content/embedding) for paper {paper_id_for_log}")
                    elif isinstance(result_or_exc, list) and result_or_exc:
                        all_chunks_to_add.extend(result_or_exc)
                        processed_papers_successfully += 1
                        logger.info(f"Successfully processed paper {paper_id_for_log}, got {len(result_or_exc)} chunks.")
                    elif isinstance(result_or_exc, list) and not result_or_exc:
                        logger.warning(f"Processing for paper {paper_id_for_log} completed but yielded no embeddable chunks.")
                        results["errors"].append(f"No embeddable chunks generated for paper {paper_id_for_log}")
                    else:
                        logger.error(f"Unexpected result type from processing task for paper {paper_id_for_log}: {type(result_or_exc)}")
                        results["errors"].append(f"Unexpected result type for paper {paper_id_for_log}: {type(result_or_exc).__name__}")

            except Exception as gather_err:
                logger.error(f"Unexpected error during asyncio.gather itself: {gather_err}", exc_info=True)
                results["message"] = f"Critical error during task gathering: {gather_err} (Traceback logged)"
                results["errors"].append(results["message"])


            # --- 8. 批量添加文献内容块到 ChromaDB ---
            added_chunks_count = 0
            if all_chunks_to_add:
                final_chunk_ids = [item['id'] for item in all_chunks_to_add]
                final_chunk_embeddings = [item['embedding'] for item in all_chunks_to_add]
                final_chunk_metadatas = [item['metadata'] for item in all_chunks_to_add]
                final_chunk_documents = [item['document'] for item in all_chunks_to_add]
                logger.info(f"Preparing to add/update {len(final_chunk_ids)} document chunks in ChromaDB collection '{db_name}'...")

                try:
                    logger.info(f"Upserting {len(final_chunk_ids)} chunks...")
                    # 使用异步线程执行 upsert
                    await asyncio.to_thread(
                        collection.upsert,
                        ids=final_chunk_ids,
                        embeddings=final_chunk_embeddings,
                        metadatas=final_chunk_metadatas,
                        documents=final_chunk_documents
                    )
                    added_chunks_count = len(final_chunk_ids)
                    results["chunks_embedded"] = added_chunks_count
                    logger.info(f"Successfully upserted {added_chunks_count} chunks into ChromaDB.")

                    # 只有在数据库操作成功后才设置 success=True 和成功消息
                    results["success"] = True
                    results["message"] = (f"PubMed found {results['papers_found']} papers. "
                                          f"Attempted to process {results['papers_processed']} papers. "
                                          f"Successfully generated and saved {added_chunks_count} content chunks "
                                          f"from {processed_papers_successfully} papers into collection '{db_name}'. "
                                          f"Combined abstracts entry saved: {results['abstracts_entry_saved']}.")

                # [修改] 明确捕获维度不匹配异常
                except chromadb.errors.InvalidDimensionException as dim_err:
                    error_msg = f"FATAL: ChromaDB upsert failed due to dimension mismatch: {dim_err}. Ensure collection dimension matches embedding dimension ({determined_embedding_dimension})."
                    logger.error(error_msg, exc_info=True)
                    results["message"] = error_msg; results["errors"].append(f"{error_msg} (Traceback logged)")
                    results["success"] = False
                    results["chunks_embedded"] = 0
                except Exception as e:
                    error_msg = f"Failed to upsert document chunks in ChromaDB: {e}"; logger.error(error_msg, exc_info=True)
                    results["message"] = f"{error_msg} (Traceback logged)"; results["errors"].append(results["message"])
                    results["success"] = False
                    results["chunks_embedded"] = 0
            else:
                logger.warning("No document chunks were successfully generated or embedded.")
                if not results["errors"] and results["papers_found"] > 0:
                    results["success"] = True
                    results["message"] = (f"Processed {results['papers_found']} papers found by PubMed, "
                                          f"but failed to obtain or embed valid content chunks for any of them. "
                                          f"Combined abstracts entry saved: {results['abstracts_entry_saved']}.")
                else:
                    results["success"] = False
                    results["message"] = (f"Processed {results['papers_found']} papers found, but encountered errors "
                                          f"and failed to generate/embed any content chunks. "
                                          f"Combined abstracts entry saved: {results['abstracts_entry_saved']}. Check errors list and logs.")


        except Exception as e: # 捕获 search_literature 函数主体中的任何未预料异常
            error_msg = f"Critical unhandled error in search_literature: {type(e).__name__} - {e}"
            logger.error(error_msg, exc_info=True)
            results["message"] = f"{error_msg}. Please check server logs for full traceback."
            results["errors"].append(results["message"])
            results["success"] = False

        finally:
            # 检查并更新最终消息
            if results["errors"] and not results["success"]:
                error_summary = "; ".join(list(set(results["errors"]))[:3]) + ('...' if len(results["errors"]) > 3 else '')
                if results["message"] and "error" not in results["message"].lower():
                    results["message"] += f" Encountered {len(results['errors'])} errors (e.g., {error_summary}). Check errors list and logs."
                elif not results["message"]:
                    results["message"] = f"Process finished with {len(results['errors'])} errors (e.g., {error_summary}). Check errors list and logs."

            end_time = time.time()
            logger.info(f"Literature search finished for '{keyword}'. Time taken: {end_time - start_time:.2f} seconds. Final status: Success={results['success']}. Message: {results['message']}")

        return results # 返回最终结果


@mcp.tool()
async def get_combined_abstracts(db_name: str) -> str:
    """
    从指定的 ChromaDB 集合中获取预存的所有摘要的合并文本（存储在 Metadata 中）。

    Args:
        db_name (str): 要查询的 ChromaDB 集合名称。

    Returns:
        str: 包含所有摘要的合并文本字符串，或者一条错误/提示信息。
    """
    logger.info(f"Attempting to retrieve combined abstracts entry ('{ALL_ABSTRACTS_DOC_ID}') from ChromaDB collection: '{db_name}'")
    combined_abstracts_text = f"未能在数据库 '{db_name}' 中找到存储的摘要合集条目 ('{ALL_ABSTRACTS_DOC_ID}') 或其内容。"

    try:
        logger.debug(f"Connecting to ChromaDB at path: {CHROMA_DB_PATH}")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        try:
            collection = await asyncio.to_thread(chroma_client.get_collection, name=db_name)
            logger.info(f"Successfully accessed collection '{db_name}'.")
        except ValueError as e:
             error_msg = f"Collection '{db_name}' does not exist at path '{CHROMA_DB_PATH}'."
             logger.error(error_msg)
             return f"错误：数据库集合 '{db_name}' 不存在。"
        except Exception as e:
            error_msg = f"Error accessing collection '{db_name}': {e}."
            logger.error(error_msg, exc_info=True)
            return f"无法访问数据库集合 '{db_name}'。错误：{e}"

        logger.info(f"Fetching entry with ID '{ALL_ABSTRACTS_DOC_ID}' from collection '{db_name}', including metadatas.")
        try:
            # [修改] 只需要获取 metadatas
            collection_data = await asyncio.to_thread(
                collection.get,
                ids=[ALL_ABSTRACTS_DOC_ID],
                include=['metadatas'] # 关键：确保包含 metadatas
            )
        except Exception as e:
             error_msg = f"Error getting entry '{ALL_ABSTRACTS_DOC_ID}' from collection '{db_name}': {e}"
             logger.error(error_msg, exc_info=True)
             return f"从数据库 '{db_name}' 获取摘要合集时出错: {e}"

        retrieved_ids = collection_data.get('ids', [])
        retrieved_metadatas = collection_data.get('metadatas') # [修改] 获取 metadatas 列表

        if ALL_ABSTRACTS_DOC_ID in retrieved_ids and retrieved_metadatas and len(retrieved_metadatas) > 0:
            metadata_dict = retrieved_metadatas[0] # 获取第一个（也是唯一一个）metadata 字典
            if metadata_dict:
                # [修改] 从 metadata 中提取我们之前存储的文本
                text_from_metadata = metadata_dict.get(COMBINED_ABSTRACT_METADATA_KEY)
                if text_from_metadata and isinstance(text_from_metadata, str):
                    combined_abstracts_text = text_from_metadata
                    logger.info(f"Successfully retrieved combined abstract text (length: {len(combined_abstracts_text)}) from metadata for ID '{ALL_ABSTRACTS_DOC_ID}'.")
                else:
                    logger.warning(f"Metadata found for ID {ALL_ABSTRACTS_DOC_ID}, but key '{COMBINED_ABSTRACT_METADATA_KEY}' is missing or not a string.")
            else:
                 logger.warning(f"Metadata list retrieved but the first element is empty/None for ID '{ALL_ABSTRACTS_DOC_ID}'.")
        else:
            logger.warning(f"Entry with ID '{ALL_ABSTRACTS_DOC_ID}' not found or has no metadata in collection '{db_name}'.")

    except Exception as e:
        error_msg = f"An unexpected error occurred while retrieving combined abstracts from '{db_name}': {e}"
        logger.error(error_msg, exc_info=True)
        combined_abstracts_text = f"处理数据库 '{db_name}' 时发生意外错误：{e}"
    finally:
        logger.info(f"Finished retrieving combined abstracts from '{db_name}'.")

    return combined_abstracts_text


@mcp.tool()
async def search_text_from_chromadb(
    db_name: str,
    reference_text: str,
    n_results: int = 5,
    delimiter: str = "\n" # <--- 新增参数，默认分隔符为换行符
) -> str:
    """
    根据提供的参考文本（按指定分隔符拆分查询），在指定的 ChromaDB 集合中搜索相关文本块。

    Args:
        db_name (str): 要查询的 ChromaDB 集合名称。
        reference_text (str): 参考文本（例如综述框架），将按 'delimiter' 参数指定的符号进行分割，
                           每个非空部分将被视为一个独立的查询。
        n_results (int): 对于参考文本的每个查询片段，希望从 ChromaDB 返回的最相似结果数量。默认为 5。
        delimiter (str): 用于分割 'reference_text' 的分隔符。默认为 '\\n' (换行符)。

    Returns:
        str: 一个包含所有查询片段及其对应搜索结果的格式化文本字符串，准备好传递给 LLM。
             如果集合不存在或发生错误，则返回错误信息。
             如果参考文本为空或分割后无有效片段，则返回提示信息。
    """
    start_time = time.time()
    logger.info(f"Starting text search in ChromaDB collection '{db_name}' based on reference text (delimiter: '{delimiter}').") # <--- 更新日志
    output_lines = [] # 用于存储最终输出的各部分

    if not reference_text or not reference_text.strip():
        logger.warning("Reference text is empty or contains only whitespace.")
        return "输入的参考文本为空，无法执行搜索。"

    # --- 1. 连接 ChromaDB (这部分代码保持不变) ---
    try:
        logger.debug(f"Connecting to ChromaDB at path: {CHROMA_DB_PATH}")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            collection = await asyncio.to_thread(chroma_client.get_collection, name=db_name)
            logger.info(f"Successfully accessed collection '{db_name}'.")
        except ValueError as e:
             error_msg = f"Collection '{db_name}' does not exist at path '{CHROMA_DB_PATH}'."
             logger.error(error_msg)
             return f"错误：数据库集合 '{db_name}' 不存在。"
        except Exception as e:
            error_msg = f"Error accessing collection '{db_name}': {e}."
            logger.error(error_msg, exc_info=True)
            return f"无法访问数据库集合 '{db_name}'。错误：{e}"

        # --- 2. 准备 Embedding (这部分代码保持不变) ---
        async with httpx.AsyncClient(timeout=60.0) as http_client:

            # --- 3. 拆分参考文本并逐个查询 ---
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # 修改点：使用 delimiter 分割，并将变量名从 lines 改为 segments
            segments = [segment.strip() for segment in reference_text.strip().split(delimiter) if segment.strip()]
            logger.info(f"Reference text split into {len(segments)} non-empty segments for querying using delimiter '{delimiter}'.")

            if not segments: # <--- 检查 segments 是否为空
                 return f"参考文本按分隔符 '{delimiter}' 拆分后没有有效的查询片段。"
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # 修改点：循环变量和日志/输出文本更新
            for i, query_segment in enumerate(segments): # 使用 segments 和 query_segment
                output_lines.append(f"\n--- 查询依据 (片段 {i+1}/{len(segments)}): \"{query_segment}\" ---") # 更新标题
                logger.info(f"Processing query segment {i+1}/{len(segments)}: \"{query_segment[:100]}...\"") # 更新日志

                # --- 3.1 获取查询嵌入 ---
                query_embedding = None
                try:
                    # 使用 query_segment 获取嵌入
                    query_embedding = await _get_single_embedding(
                        query_segment, # <--- 使用 query_segment
                        SILICONFLOW_API_KEY,
                        SILICONFLOW_API_URL,
                        SILICONFLOW_EMBEDDING_MODEL,
                        http_client
                    )
                    if query_embedding is None:
                        logger.error(f"Failed to get embedding for query segment: \"{query_segment[:100]}...\"") # 更新日志
                        output_lines.append("  [错误：无法生成此片段的查询向量]")
                        continue
                    logger.debug(f"Successfully obtained embedding for query segment {i+1}.") # 更新日志

                except Exception as embed_err:
                    logger.error(f"Error getting embedding for query segment \"{query_segment[:100]}...\": {embed_err}", exc_info=True) # 更新日志
                    output_lines.append(f"  [错误：生成查询向量时出错: {embed_err}]")
                    continue

                # --- 3.2 执行 ChromaDB 查询 ---
                try:
                    logger.debug(f"Querying ChromaDB collection '{db_name}' with embedding for segment {i+1}, n_results={n_results}.") # 更新日志
                    query_results = await asyncio.to_thread(
                        collection.query,
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        include=['metadatas', 'documents', 'distances']
                    )
                    logger.debug(f"ChromaDB query completed for segment {i+1}.") # 更新日志

                except Exception as query_err:
                    logger.error(f"Error querying ChromaDB for segment \"{query_segment[:100]}...\": {query_err}", exc_info=True) # 更新日志
                    output_lines.append(f"  [错误：在数据库中查询时出错: {query_err}]")
                    continue
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                # --- 3.3 格式化查询结果 (这部分代码保持不变) ---
                ids_list = query_results.get('ids', [[]])[0]
                distances_list = query_results.get('distances', [[]])[0]
                metadatas_list = query_results.get('metadatas', [[]])[0]
                documents_list = query_results.get('documents', [[]])[0]

                if not ids_list:
                    logger.info(f"No relevant documents found in '{db_name}' for query segment: \"{query_segment[:100]}...\"") # 更新日志
                    output_lines.append("  [未找到相关内容]")
                    continue

                output_lines.append(f"  找到 {len(ids_list)} 个相关内容块:")
                for rank, (doc_id, distance, metadata, document) in enumerate(zip(ids_list, distances_list, metadatas_list, documents_list), 1):
                    title = metadata.get('title', 'N/A')
                    source_info = metadata.get('source', 'N/A')
                    original_id = metadata.get('original_paper_id', 'N/A')
                    chunk_idx = metadata.get('chunk_index', 'N/A')
                    source_details = f"来源: {source_info}"
                    if title != 'N/A': source_details += f", 标题: {title[:80]}..."
                    if original_id != 'N/A': source_details += f", 原文ID: {original_id}"
                    if chunk_idx != 'N/A': source_details += f", 块索引: {chunk_idx}"
                    snippet = document.strip()
                    max_snippet_len = 300
                    if len(snippet) > max_snippet_len:
                        snippet = snippet[:max_snippet_len] + "..."
                    output_lines.append(f"    {rank}. 相关度 (距离): {distance:.4f}")
                    output_lines.append(f"       {source_details}")
                    output_lines.append(f"       文本片段: \"{snippet}\"")
                    output_lines.append("-" * 15)

            # --- 4. 组合最终输出 (这部分代码保持不变) ---
            final_output = "\n".join(output_lines)

    except Exception as e:
        # ... (异常处理保持不变) ...
        error_msg = f"An unexpected error occurred during search_text_from_chromadb for collection '{db_name}': {e}"
        logger.error(error_msg, exc_info=True)
        return f"执行文本搜索时发生意外错误: {e}"

    finally:
        # ... (结束日志保持不变) ...
        end_time = time.time()
        logger.info(f"Finished text search in ChromaDB '{db_name}'. Time taken: {end_time - start_time:.2f} seconds.")

    return final_output.strip() # 返回组合后的文本


# [修改] 调整辅助函数签名以接收共享的 httpx.AsyncClient
async def fetch_content_and_process_chunks(paper_input: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    获取单篇论文的内容（优先PMC，然后PDF，最后摘要），进行分块和嵌入。
    限制并发 embedding 请求数量，并使用共享的 httpx client。
    """
    original_paper_id = paper_input["original_paper_id"]
    doi = paper_input.get("doi")
    abstract = paper_input.get("abstract")
    pmc_text = paper_input.get("pmc_text")
    initial_source = paper_input["initial_source"]
    title = paper_input.get("title", "N/A")
    # [修改] 获取共享的 client
    http_client = paper_input["http_client"]
    text_to_embed = None
    final_source = "N/A"

    # 1. 确定用于嵌入的文本内容 (逻辑不变)
    if pmc_text:
        text_to_embed = pmc_text; final_source = initial_source
        logger.info(f"Using PMC text for embedding paper {original_paper_id}.")
    else:
        if doi:
            logger.info(f"PMC text not available for {original_paper_id}, trying Sci-Hub PDF via DOI: {doi}")
            pdf_url = await asyncio.to_thread(run_scihub_web_request, doi)
            if pdf_url:
                # 注意：_get_pdf_text_from_url_internal 内部也应使用异步 client，这里假设它已修改
                # 或者在 _get_pdf_text_from_url_internal 内部创建临时的 client
                pdf_text = await _get_pdf_text_from_url_internal(pdf_url) # 确保此函数也使用异步 client
                if pdf_text:
                    text_to_embed = pdf_text; final_source = "Sci-Hub PDF"
                    logger.info(f"Using Sci-Hub PDF text for embedding paper {original_paper_id}.")
                else: logger.warning(f"Sci-Hub PDF extraction failed or timed out for {original_paper_id}. Will try abstract.")
            else: logger.info(f"Sci-Hub link not found for {original_paper_id}. Will try abstract.")
        if text_to_embed is None:
            if abstract:
                text_to_embed = abstract; final_source = "Abstract"
                logger.info(f"Using abstract for embedding paper {original_paper_id}.")
            else:
                logger.warning(f"No content (PMC, PDF, or Abstract) found for paper {original_paper_id}. Cannot process chunks.")
                return None

    # 2. 分块和嵌入
    if not text_to_embed or not text_to_embed.strip():
        logger.warning(f"Content for paper {original_paper_id} (Source: {final_source}) is empty. Skipping.")
        return None
    logger.info(f"Processing content chunks for paper {original_paper_id} from source: {final_source}. Original text length: {len(text_to_embed)}")
    chunks = split_text_into_chunks(text_to_embed, CONTENT_CHUNK_SIZE, CONTENT_CHUNK_OVERLAP)
    if not chunks:
        logger.warning(f"Text splitting resulted in zero chunks for paper {original_paper_id}. Skipping.")
        return None
    logger.info(f"Split content into {len(chunks)} chunks for paper {original_paper_id}.")

    chunks_data_list = []
    embedding_tasks = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMBEDDING_REQUESTS)
    REQUEST_DELAY = 1

    async def get_embedding_with_semaphore(chunk_text: str) -> Optional[List[float]]:
        async with semaphore:
            logger.debug(f"Waiting {REQUEST_DELAY}s before next embedding request...") # 添加日志方便观察
            await asyncio.sleep(REQUEST_DELAY) # 使用调整后的延迟时间
            return await _get_single_embedding(
                chunk_text, SILICONFLOW_API_KEY, SILICONFLOW_API_URL, SILICONFLOW_EMBEDDING_MODEL, http_client
            )

    for i, chunk_text in enumerate(chunks):
        embedding_tasks.append(get_embedding_with_semaphore(chunk_text))

    chunk_embeddings = await asyncio.gather(*embedding_tasks, return_exceptions=True)

    successful_embeddings_count = 0
    for i, embedding_or_exc in enumerate(chunk_embeddings):
        chunk_text = chunks[i]
        chunk_id = f"{original_paper_id}_chunk_{i}"
        if isinstance(embedding_or_exc, Exception):
            if isinstance(embedding_or_exc, httpx.ConnectError):
                 logger.error(f"ConnectError getting embedding for chunk {i} of paper {original_paper_id}: {embedding_or_exc}", exc_info=False)
            else:
                 logger.error(f"Failed to get embedding for chunk {i} of paper {original_paper_id}: {embedding_or_exc}", exc_info=True)
        elif embedding_or_exc is None:
            logger.warning(f"Embedding returned None for chunk {i} of paper {original_paper_id}. Skipping this chunk.")
        elif isinstance(embedding_or_exc, list):
            # 检查维度（可选但建议）
            # if determined_embedding_dimension and len(embedding_or_exc) != determined_embedding_dimension:
            #     logger.error(f"Chunk {i} embedding dim ({len(embedding_or_exc)}) mismatch expected ({determined_embedding_dimension}) for paper {original_paper_id}. Skipping.")
            #     continue # 跳过维度不匹配的块
            chunk_metadata = {
                "title": title,
                "doi": doi,
                "original_paper_id": original_paper_id,
                "chunk_index": i,
                "source": final_source
            }
            chunk_metadata = {k: v for k, v in chunk_metadata.items() if v is not None}
            chunks_data_list.append({
                "id": chunk_id,
                "embedding": embedding_or_exc,
                "metadata": chunk_metadata,
                "document": chunk_text
            })
            successful_embeddings_count += 1
        else:
             logger.error(f"Unexpected return type from embedding task for chunk {i} of paper {original_paper_id}: {type(embedding_or_exc)}")

    if not chunks_data_list:
        logger.warning(f"Failed to embed any chunks for paper {original_paper_id}.")
        return None
    logger.info(f"Successfully embedded {successful_embeddings_count} out of {len(chunks)} chunks for paper {original_paper_id}.")
    return chunks_data_list


# --- 主程序入口 ---
if __name__ == "__main__":
    logger.info("Starting Literature Search MCP Server (via stdio)...")
    try:
        if not os.path.exists(CHROMA_DB_PATH):
             logger.info(f"ChromaDB path '{CHROMA_DB_PATH}' does not exist. Attempting to create it.")
             os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        else:
             logger.info(f"ChromaDB path '{CHROMA_DB_PATH}' already exists.")
        test_file = os.path.join(CHROMA_DB_PATH, ".write_test")
        with open(test_file, "w") as f: f.write("test")
        os.remove(test_file)
        logger.info(f"Successfully tested write access to ChromaDB path: {CHROMA_DB_PATH}")
    except OSError as e:
        logger.error(f"Error accessing or creating ChromaDB path '{CHROMA_DB_PATH}': {e}. Check permissions.", exc_info=True)
        sys.exit(1)
    except Exception as e:
         logger.error(f"Unexpected error during ChromaDB path check '{CHROMA_DB_PATH}': {e}.", exc_info=True)
         # 视情况决定是否退出

    mcp.run()
    logger.info("MCP Server stopped.")

