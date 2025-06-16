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
import shutil # [新增] 用于移动文件
import json # [新增] 用于处理摘要列表
import chromadb
import numpy as np # 用于向量计算
import re 
# import numpy as np # 暂时不需要 numpy
# import PyPDF2 # PyPDF2 现在由 marker 替代，暂时注释掉，但函数保留以防万一
import requests
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
import httpx # 显式导入 httpx
# from tenacity import retry, stop_after_attempt, wait_exponential # 可以考虑添加重试库
# --- 配置 ---
# PubMed API
NCBI_EMAIL = "your.email@example.com" #设置 替换成你的ncbi注册邮箱，非必须
NCBI_API_KEY = "xxxxx" #设置 替换为你的ncbi_api，可选
# 硅基流动 Embedding API
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/embeddings"
# !!! 安全警告：切勿在生产代码中硬编码 API Key !!!
# 建议使用环境变量或其他安全方式管理
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-xxxxx") #设置 #将此处”sk-xxxx“替换为你的硅基流动的真实 Key 
SILICONFLOW_EMBEDDING_MODEL = "Pro/BAAI/bge-m3"
MAX_CONCURRENT_EMBEDDING_REQUESTS = 3 # 并发 Embedding 请求限制
EMBEDDING_REQUEST_DELAY = 0.5 # 每个 Embedding 请求前的延迟（秒）
# Embedding 模型配置
SEMANTIC_WINDOW_SIZE = 3         # 滑动窗口大小（句子数）
SEMANTIC_SIMILARITY_THRESHOLD = 0.35 # 相邻句子语义相似度阈值 (需要根据效果调整)
MIN_SEMANTIC_CHUNK_SENTENCES = 2 # 一个语义块最少包含的句子数，防止切分过细
CONTENT_CHUNK_SIZE = 2048      # 文献内容分块的目标字符数
CONTENT_CHUNK_OVERLAP = 128    # 文献内容分块之间的重叠字符数
MAX_CHARS_FOR_EMBEDDING_API_LIMIT = 15000 # 保守估计的模型单次输入字符上限 (用于内部检查)
SILICONFLOW_EMBEDDING_BATCH_SIZE = 64
# ChromaDB
CHROMA_DB_PATH = "xxxx" #设置 为你的本地数据库保存路径
DEFAULT_RETMAX_PUBMED = 20 # 默认检索数量
PDF_PARSE_TIMEOUT = 120.0 # PDF 解析超时时间（秒）- [注意] 此项现在不直接使用，由 marker 控制
DOWNLOAD_TIMEOUT = 60.0 # 文件下载超时时间（秒）
ALL_ABSTRACTS_DOC_ID = "doc_all_abstracts" # 用于存储所有摘要整合文档的固定 ID
COMBINED_ABSTRACT_METADATA_KEY = "pmid_abstract_pairs_json" # [修改] 存储 pmid+摘要 json 的 metadata key
# Sci-Hub
SCIHUB_MIRRORS = [
    "https://sci-hub.se",
    "https://sci-hub.st",
    "https://sci-hub.ru",
    # 可以添加更多镜像
]
# Marker 配置
MARKER_WORKERS = 4 #设置 marker 使用的 worker 数量，如果电脑性能差建议选择为1
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
def make_request_with_retry(url: str, params: Dict[str, Any], method: str = 'get', max_retries: int = 3, wait_time: float = 1.0, timeout: float = 30.0) -> requests.Response:
    """带重试的 HTTP 请求 (使用 requests)"""
    effective_params = params.copy()
    if NCBI_EMAIL and NCBI_EMAIL != "your.email@example.com":
        effective_params['email'] = NCBI_EMAIL
    if NCBI_API_KEY:
        effective_params['api_key'] = NCBI_API_KEY
    response = None
    for attempt in range(max_retries):
        try:
            logger.debug(f"Making {method.upper()} request to {url} (attempt {attempt + 1})")
            if method.lower() == 'get':
                response = requests.get(url, params=effective_params, timeout=timeout)
            elif method.lower() == 'post':
                 # POST 请求的超时可以设置得长一些
                 post_timeout = 60.0 if timeout <= 30.0 else timeout
                 response = requests.post(url, data=effective_params, timeout=post_timeout)
            else: raise ValueError(f"Unsupported HTTP method: {method}")
            response.raise_for_status()
            logger.debug(f"Request successful (Status: {response.status_code})")
            return response
        except requests.exceptions.Timeout as e:
             logger.warning(f"Request timed out (attempt {attempt + 1}/{max_retries}) for {url}: {str(e)}")
             if attempt == max_retries - 1: raise
             time.sleep(wait_time); wait_time *= 2
        except requests.exceptions.RequestException as e:
            status_code = response.status_code if response is not None else 'N/A'
            logger.error(f"Request failed (attempt {attempt + 1}/{max_retries}) for {url} (Status: {status_code}): {str(e)}")
            if response is not None and 400 <= response.status_code < 500 and response.status_code != 429: # 429 Too Many Requests 可以重试
                logger.error(f"Client error ({response.status_code}), not retrying.")
                raise
            if attempt == max_retries - 1: raise
            time.sleep(wait_time); wait_time *= 2
    raise Exception(f"Request failed after {max_retries} retries for {url}")
def parse_pubmed_xml_details(xml_content: bytes) -> List[Dict[str, Any]]:
    """解析 PubMed EFetch XML 获取元数据 (含 DOI, PMCID, 完整摘要)"""
    results = []
    if not xml_content: return results
    try:
        root = ET.fromstring(xml_content)
        for article in root.findall('.//PubmedArticle'):
            entry = {'pmid': None, 'title': 'N/A', 'abstract': None, 'doi': None, 'pmcid': None}
            pmid_elem = article.find('.//PMID')
            if pmid_elem is not None and pmid_elem.text: entry['pmid'] = pmid_elem.text.strip()
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
            article_id_list_node = article.find('.//PubmedArticleSet/PubmedArticle/PubmedData/ArticleIdList') # 更精确路径
            if article_id_list_node is None: # 尝试备用路径
                 article_id_list_node = article.find('.//ArticleIdList')
            if article_id_list_node is not None:
                for item in article_id_list_node.findall('./ArticleId'):
                    id_type = item.get('IdType')
                    if id_type == 'doi' and item.text: doi = item.text.strip()
                    elif id_type == 'pmc' and item.text:
                         raw_pmcid = item.text.strip()
                         # 确保 PMCID 以 "PMC" 开头
                         if raw_pmcid.isdigit(): pmcid = f"PMC{raw_pmcid}"
                         elif raw_pmcid.startswith("PMC"): pmcid = raw_pmcid
                         else: logger.warning(f"Found unusual PMCID format: {raw_pmcid} for PMID {entry['pmid']}")
            # 有些 DOI 可能在 ELocationID 中
            if not doi:
                 doi_elem = article.find(".//ELocationID[@EIdType='doi']")
                 if doi_elem is not None and doi_elem.text: doi = doi_elem.text.strip()
            entry['doi'] = doi
            entry['pmcid'] = pmcid
            if entry['pmid']: results.append(entry)
            else: logger.warning(f"Skipping article due to missing PMID. Title: {entry['title']}")
    except ET.ParseError as e: logger.error(f"Failed to parse PubMed XML: {e}", exc_info=True); raise
    except Exception as e: logger.error(f"Error processing PubMed XML: {e}", exc_info=True); raise
    return results
# parse_pmc_full_text_xml 暂时保留，以防将来需要 XML 内容
def parse_pmc_full_text_xml(xml_content: bytes, requested_pmcids: List[str]) -> Dict[str, str]:
    """解析 PMC 全文 XML (此函数在新流程中可能较少使用)"""
    # ... (保持原样) ...
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
    """内部函数：搜索 PubMed，支持排序和日期过滤。"""
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
        # 使用异步线程执行 requests 调用
        search_response = await asyncio.to_thread(make_request_with_retry, search_url, search_params)
        search_data = search_response.json()
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids: logger.info("No PMIDs found."); return []
        logger.info(f"Found {len(pmids)} PMIDs.")
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {'db': 'pubmed', 'id': ",".join(pmids), 'retmode': 'xml', 'rettype': 'abstract'}
        # 决定使用 GET 还是 POST
        use_post = len(",".join(pmids)) > 1000 or len(pmids) > 150 # URL 长度或 ID 数量限制
        fetch_method = 'post' if use_post else 'get'
        fetch_response = await asyncio.to_thread(make_request_with_retry, fetch_url, fetch_params, method=fetch_method, timeout=60.0) # 增加 EFetch 超时
        articles = await asyncio.to_thread(parse_pubmed_xml_details, fetch_response.content)
        logger.info(f"Successfully parsed metadata for {len(articles)} articles.")
    except Exception as e:
        logger.error(f"Error during PubMed search: {e}", exc_info=True)
        # 可以在这里根据需要重新抛出异常或返回空列表
    return articles
# _get_pmc_full_text_internal 暂时保留，以防万一
async def _get_pmc_full_text_internal(pmcids: List[str]) -> Dict[str, str]:
    """内部函数：获取 PMC 全文 XML (在新流程中可能较少使用)"""
    # ... (保持原样) ...
    if not pmcids: return {}
    logger.info(f"[Deprecated path] Fetching PMC full text XML for {len(pmcids)} PMCID(s): {pmcids}")
    full_texts = {pmcid: f"Error fetching full text for {pmcid}." for pmcid in pmcids}
    try:
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {'db': 'pmc', 'id': ",".join(pmcids), 'retmode': 'xml', 'rettype': 'full'}
        fetch_method = 'post' if len(pmcids) > 50 else 'get' # 简化判断
        fetch_response = await asyncio.to_thread(make_request_with_retry, fetch_url, fetch_params, method=fetch_method, timeout=120.0) # 增加超时
        parsed_texts = await asyncio.to_thread(parse_pmc_full_text_xml, fetch_response.content, pmcids)
        return parsed_texts
    except Exception as e:
        logger.error(f"Error fetching PMC full text XML: {e}", exc_info=True)
        error_msg = f"Error fetching full text XML: {e}"
        for pmcid in pmcids: full_texts[pmcid] = error_msg
    return full_texts
# --- 新增：PMC HTML 页面抓取辅助函数 ---
async def scrape_pmc_for_pdf_link(pmc_article_url: str, client: httpx.AsyncClient) -> Optional[str]:
    """尝试从 PMC 文章页面 HTML 中提取 PDF 下载链接。"""
    # ... (保持不变) ...
    logger.info(f"Scraping PMC page for PDF link: {pmc_article_url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = await client.get(pmc_article_url, headers=headers, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '').lower()
        if 'html' not in content_type:
             logger.warning(f"Unexpected Content-Type '{content_type}' for PMC page {pmc_article_url}. Skipping scrape.")
             return None
        html_content = response.text
        soup = BeautifulSoup(html_content, 'lxml')
        meta_tag = soup.find('meta', attrs={'name': 'citation_pdf_url'})
        if meta_tag and meta_tag.get('content'):
            pdf_url = meta_tag['content'].strip()
            if pdf_url.startswith('//'): pdf_url = 'https:' + pdf_url
            elif pdf_url.startswith('/'):
                 base_url = '/'.join(pmc_article_url.split('/')[:3])
                 pdf_url = base_url + pdf_url
            logger.info(f"Found PDF link via PMC HTML meta tag: {pdf_url}")
            return pdf_url
        else:
            pdf_link_tag = soup.find('a', class_=re.compile(r'format-pdf|pdf-link', re.I), href=re.compile(r'\.pdf', re.I))
            if pdf_link_tag and pdf_link_tag.get('href'):
                pdf_url = pdf_link_tag['href'].strip()
                if pdf_url.startswith('//'): pdf_url = 'https:' + pdf_url
                elif pdf_url.startswith('/'):
                    base_url = '/'.join(pmc_article_url.split('/')[:3])
                    pdf_url = base_url + pdf_url
                elif not pdf_url.startswith('http'):
                     from urllib.parse import urljoin
                     pdf_url = urljoin(pmc_article_url, pdf_url)
                logger.info(f"Found PDF link via PMC HTML fallback (<a> tag): {pdf_url}")
                return pdf_url
            logger.info(f"PDF link meta tag ('citation_pdf_url') and common fallback links not found on PMC page: {pmc_article_url}")
            return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP error {e.response.status_code} accessing PMC page {pmc_article_url}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Request error accessing PMC page {pmc_article_url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error scraping PMC page {pmc_article_url}: {e}", exc_info=True)
        return None
# --- Sci-Hub & PDF 辅助函数 (部分修改/新增) ---
# run_scihub_web_request 保持不变
def run_scihub_web_request(doi: str) -> Optional[str]:
    """尝试从 Sci-Hub 镜像获取 PDF 链接"""
    # ... (保持不变) ...
    for mirror in SCIHUB_MIRRORS:
        try:
            scihub_url = f"{mirror}/{doi}"
            logger.info(f"Trying Sci-Hub mirror: {scihub_url}")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(scihub_url, timeout=20, headers=headers, allow_redirects=True)
            response.raise_for_status()
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'lxml')
                pdf_url = None
                embed = soup.find('embed', type='application/pdf')
                if embed and embed.get('src'): pdf_url = embed['src']
                if not pdf_url:
                    iframe = soup.find('iframe', id='pdf')
                    if iframe and iframe.get('src'): pdf_url = iframe.get('src')
                    else:
                        iframes = soup.find_all('iframe')
                        for frame in iframes:
                            src = frame.get('src', '')
                            if src and 'pdf' in src.lower(): pdf_url = src; break
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
                         pdf_url = base_url + '/' + pdf_url.lstrip('/')
                    logger.info(f"Found PDF link via Sci-Hub ({mirror}): {pdf_url}")
                    return pdf_url
                elif "article not found" in response.text.lower() or "article unavailable" in response.text.lower():
                     logger.info(f"Article not found/unavailable on Sci-Hub mirror {mirror} for DOI {doi}")
                     return None
                else: logger.warning(f"Could not find common PDF link structures on page {scihub_url} (mirror: {mirror})")
            else:
                logger.warning(f"Sci-Hub mirror {mirror} returned non-200 status: {response.status_code} for DOI {doi}")
        except requests.exceptions.Timeout:
             logger.warning(f"Timeout accessing Sci-Hub mirror {mirror} for DOI {doi}")
        except requests.exceptions.HTTPError as http_err:
             logger.warning(f"HTTP error accessing Sci-Hub mirror {mirror} for DOI {doi}: {http_err}")
             if http_err.response.status_code == 404:
                  logger.info(f"Article likely not found (404) on Sci-Hub mirror {mirror} for DOI {doi}")
        except Exception as e:
             logger.error(f"Error processing Sci-Hub for DOI {doi} from {mirror}: {str(e)}", exc_info=True)
        time.sleep(0.5)
    logger.warning(f"Failed to get PDF link from all Sci-Hub mirrors for DOI: {doi}")
    return None
# [新增] 文件下载辅助函数
async def download_file(url: str, local_path: str, client: httpx.AsyncClient) -> bool:
    """使用 httpx 流式下载文件到本地。"""
    # ... (保持不变) ...
    logger.info(f"Attempting to download file from {url} to {local_path}")
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        headers = {"User-Agent": "Mozilla/5.0"}
        async with client.stream("GET", url, headers=headers, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as response:
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').lower()
            is_pdf = 'application/pdf' in content_type
            is_octet = 'application/octet-stream' in content_type
            if not is_pdf and not is_octet:
                logger.warning(f"URL {url} Content-Type ('{content_type}') might not be a standard PDF type, but download will proceed.")
            if 'text/html' in content_type:
                 logger.warning(f"URL {url} returned HTML content. This might be an error page instead of the PDF. Download will proceed but may fail later.")
            try:
                with open(local_path, "wb") as f:
                    bytes_downloaded = 0
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                    logger.info(f"Successfully downloaded {bytes_downloaded} bytes to {local_path}")
                if bytes_downloaded < 100:
                    logger.warning(f"Downloaded file {local_path} seems very small ({bytes_downloaded} bytes). It might be invalid.")
                return True
            except IOError as e:
                logger.error(f"IOError writing file {local_path}: {e}", exc_info=True)
                if os.path.exists(local_path):
                    try: os.remove(local_path)
                    except OSError: pass
                return False
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404: logger.warning(f"File not found (404) at {url}")
        else: logger.error(f"HTTP error downloading file from {url}: {e.response.status_code}")
        return False
    except httpx.TimeoutException:
        logger.error(f"Timeout downloading file from {url} after {DOWNLOAD_TIMEOUT} seconds.")
        return False
    except httpx.RequestError as e:
         logger.error(f"Failed to download file from {url}: {type(e).__name__} - {str(e)}")
         return False
    except Exception as e:
         logger.error(f"Unexpected error downloading file from {url}: {str(e)}", exc_info=True)
         return False
# [保留] PDF 文本提取函数，当前流程不直接使用，但保留代码
# import PyPDF2 # 移到函数内部，或在顶部注释掉
def extract_text_from_local_pdf(pdf_path: str) -> Optional[str]:
    """从本地 PDF 文件中提取文本 (当前流程不直接使用)"""
    import PyPDF2 # 仅在此函数需要时导入
    logger.info(f"Extracting text from local PDF file: {pdf_path}")
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found at path: {pdf_path}")
        return None
    try:
        with open(pdf_path, 'rb') as pdf_stream:
            reader = PyPDF2.PdfReader(pdf_stream, strict=False)
            if reader.is_encrypted:
                logger.warning(f"PDF {pdf_path} is encrypted. Attempting to decrypt with default password.")
                try:
                    if reader.decrypt('') == 0:
                         logger.error(f"Failed to decrypt PDF {pdf_path} with empty password.")
                         return None
                    else: logger.info(f"PDF {pdf_path} successfully decrypted.")
                except Exception as decrypt_err:
                     logger.error(f"Error during decryption attempt for PDF {pdf_path}: {decrypt_err}")
                     return None
            num_pages = len(reader.pages)
            if num_pages == 0:
                 logger.warning(f"PDF {pdf_path} contains 0 pages.")
                 return None
            logger.debug(f"PDF {pdf_path} has {num_pages} pages.")
            text_parts = []
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        page_text = ''.join(c for c in page_text if c.isprintable() or c.isspace())
                        text_parts.append(page_text.strip())
                except Exception as page_err:
                    logger.warning(f"Error extracting text from page {i+1} of PDF {pdf_path}: {page_err}")
                    continue
            if not text_parts:
                 logger.warning(f"No text could be extracted from any page of PDF: {pdf_path}")
                 return None
            full_text = "\n\n".join(text_parts)
            full_text = re.sub(r'\s*\n\s*', '\n', full_text)
            full_text = re.sub(r'\n{3,}', '\n\n', full_text)
            full_text = re.sub(r'[ \t]{2,}', ' ', full_text)
            full_text = re.sub(r'-\n', '', full_text)
            full_text = full_text.strip()
            if full_text:
                 logger.info(f"Successfully extracted text (approx. {len(full_text)} chars) from PDF: {pdf_path}")
                 return full_text
            else:
                 logger.warning(f"Extracted empty text after processing PDF: {pdf_path}")
                 return None
    except PyPDF2.errors.PdfReadError as pdf_err:
        logger.error(f"PyPDF2 error reading PDF {pdf_path}: {pdf_err}. File might be corrupted or unsupported.")
        return None
    except FileNotFoundError:
        logger.error(f"PDF file disappeared before processing: {pdf_path}")
        return None
    except Exception as e:
         logger.error(f"Unexpected error extracting text from PDF {pdf_path}: {str(e)}", exc_info=True)
         return None
# --- Embedding 辅助函数 ---
# split_text_into_chunks 保持不变
def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """将长文本按目标字符数分割成带重叠的块"""
    # ... (保持不变) ...
    if not text or not text.strip(): return []
    text = text.strip(); text_len = len(text)
    if text_len <= chunk_size: return [text]
    chunks = []; start_index = 0
    while start_index < text_len:
        end_index = start_index + chunk_size; chunk = text[start_index:end_index]
        if end_index < text_len:
            best_break = -1
            m = re.search(r'[.?!]\s', chunk[chunk_size-overlap:])
            if m: best_break = chunk_size - overlap + m.end()
            else:
                 nl = chunk.rfind('\n', chunk_size - overlap)
                 if nl != -1: best_break = nl + 1
                 else:
                      sp = chunk.rfind(' ', chunk_size - overlap)
                      if sp != -1: best_break = sp + 1
            if best_break != -1 and best_break > chunk_size - overlap:
                end_index = start_index + best_break
                chunk = text[start_index:end_index]
        chunks.append(chunk.strip())
        next_start = end_index - overlap
        if next_start <= start_index: next_start = start_index + 1
        start_index = next_start
    return [c for c in chunks if c]
# _get_single_embedding 保持不变


logger = logging.getLogger(__name__)
def paragraph_based_split_text(text: str, max_chunk_chars: int = 15000, min_chunk_chars: int = 100) -> List[str]:
    """
    基于段落和标点符号分割文本。
    1. 按段落分割。
    2. 短段落 (<= max_chunk_chars) 直接作为块。
    3. 长段落 (> max_chunk_chars) 按句号分割，然后重组片段以适应 max_chunk_chars。
    4. 舍弃长度 < min_chunk_chars 的块。
    """
    final_chunks = []
    if not text or not text.strip():
        return []
    # 1. 按段落分割 (匹配一个或多个换行符，中间可有空白)
    paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    for i, paragraph in enumerate(paragraphs):
        if not paragraph: # 跳过空段落
            continue
        if len(paragraph) <= max_chunk_chars:
            # 2. 如果段落长度小于等于 max_chunk_chars，它本身就是一个文本块
            if len(paragraph) >= min_chunk_chars:
                final_chunks.append(paragraph)
            # else: (如果段落本身 < min_chunk_chars，则舍弃)
            #    logger.debug(f"Paragraph {i+1} (length {len(paragraph)}) is shorter than min_chunk_chars ({min_chunk_chars}), discarding.")
        else:
            # 3. 如果段落长度大于 max_chunk_chars，则需要进一步处理
            # 根据标点符号 . (英文句号) 来分割这个长段落
            # (?<=\.) 表示在句号之后分割，并保留句号在前面的部分; \s* 匹配句号后的任意空格
            sentences = re.split(r'(?<=\.)\s*', paragraph)
            sentences = [s.strip() for s in sentences if s.strip()]
            current_sentences_for_chunk = []
            current_length = 0
            for sent_idx, sentence in enumerate(sentences):
                if not sentence:
                    continue
                # 如果单个句子就超长，则记录警告并跳过 (根据需求，不能超过15000)
                if len(sentence) > max_chunk_chars:
                    logger.warning(
                        f"Paragraph {i+1}, sentence {sent_idx+1} (length {len(sentence)}) "
                        f"exceeds max_chunk_chars ({max_chunk_chars}). Skipping this sentence."
                    )
                    # 如果之前有累积的句子，先处理它们
                    if current_sentences_for_chunk:
                        combined_chunk = " ".join(current_sentences_for_chunk).strip()
                        if len(combined_chunk) >= min_chunk_chars:
                            final_chunks.append(combined_chunk)
                        current_sentences_for_chunk = []
                        current_length = 0
                    continue
                # 尝试将当前句子加入到正在构建的块中
                # 计算加入后的长度 (如果不是第一个句子，则加一个空格的长度)
                sentence_len_with_space = len(sentence) + (1 if current_sentences_for_chunk else 0)
                if current_length + sentence_len_with_space <= max_chunk_chars:
                    current_sentences_for_chunk.append(sentence)
                    current_length += sentence_len_with_space
                else:
                    # 添加此句子会超长，因此先保存当前累积的块
                    if current_sentences_for_chunk:
                        combined_chunk = " ".join(current_sentences_for_chunk).strip()
                        if len(combined_chunk) >= min_chunk_chars:
                            final_chunks.append(combined_chunk)
                    # 当前句子成为新块的开始
                    current_sentences_for_chunk = [sentence]
                    current_length = len(sentence) # 新块的长度是当前句子的长度
            # 处理最后一个累积的块
            if current_sentences_for_chunk:
                combined_chunk = " ".join(current_sentences_for_chunk).strip()
                if len(combined_chunk) >= min_chunk_chars:
                    final_chunks.append(combined_chunk)
    
    # 最终确认所有块都符合 min_chunk_chars (理论上在添加时已检查，但可作为保险)
    # final_chunks = [chunk for chunk in final_chunks if len(chunk) >= min_chunk_chars]
    return final_chunks

# [新增] 批量获取嵌入的函数
async def _get_embeddings_batch(
    texts: List[str],
    api_key: str,
    api_url: str,
    model: str,
    client: httpx.AsyncClient,
    batch_size: int = SILICONFLOW_EMBEDDING_BATCH_SIZE # 使用常量
) -> List[Optional[List[float]]]:
    """
    [新增] 使用 API 批量获取文本列表的嵌入。
    处理 API 的批量大小限制，并包含重试逻辑。
    返回与输入文本列表顺序一致的嵌入列表，失败则为 None。
    """
    if not texts:
        return []
    all_embeddings: List[Optional[List[float]]] = [None] * len(texts) # 初始化结果列表
    tasks_to_process = [] # 存储需要执行的批量 API 调用任务
    # 将文本列表分割成符合 API 限制的批次
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:min(i + batch_size, len(texts))]
        if not batch_texts: continue # 跳过空批次
        # 为每个批次创建一个处理任务
        async def process_single_batch(current_batch_texts: List[str], start_index: int):
            logger.debug(f"Processing embedding batch starting at index {start_index}, size {len(current_batch_texts)}...")
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            # 检查并截断批次中过长的文本
            truncated_batch_texts = []
            original_indices = [] # 记录原始文本在批次中的索引
            for idx, text in enumerate(current_batch_texts):
                if not text or not text.strip():
                    logger.warning(f"Skipping embedding for empty text at batch index {idx} (original index {start_index + idx}).")
                    truncated_batch_texts.append(None) # 使用 None 占位
                    original_indices.append(idx)
                    continue
                if len(text) > MAX_CHARS_FOR_EMBEDDING_API_LIMIT:
                    logger.warning(f"Text length ({len(text)}) at batch index {idx} (original index {start_index + idx}) exceeds API limit ({MAX_CHARS_FOR_EMBEDDING_API_LIMIT}). Truncating.")
                    truncated_batch_texts.append(text[:MAX_CHARS_FOR_EMBEDDING_API_LIMIT])
                else:
                    truncated_batch_texts.append(text)
                original_indices.append(idx)
            # 过滤掉 None (空文本)，只发送有效文本到 API
            valid_texts_in_batch = [t for t in truncated_batch_texts if t is not None]
            valid_original_indices = [original_indices[i] for i, t in enumerate(truncated_batch_texts) if t is not None]
            if not valid_texts_in_batch: # 如果整个批次都是空文本
                 logger.warning(f"Entire batch starting at index {start_index} consists of empty texts. Skipping API call.")
                 return # 这个批次没有结果
            payload = {"input": valid_texts_in_batch, "model": model}
            retry_count = 0; max_retries = 2; wait_time = 1.0
            batch_results: Optional[Dict[int, List[float]]] = None # 存储批次结果 {原始批次索引: embedding}
            while retry_count <= max_retries:
                try:
                    # 在这里可以加入速率限制器 (如果实现了)
                    # async with api_limiter:
                    response = await client.post(api_url, headers=headers, json=payload, timeout=120.0) # 增加批量请求超时
                    response.raise_for_status()
                    response_data = response.json()
                    if "data" in response_data and isinstance(response_data["data"], list):
                        batch_results = {}
                        api_embeddings = response_data["data"]
                        if len(api_embeddings) != len(valid_texts_in_batch):
                             logger.error(f"API returned {len(api_embeddings)} embeddings for a batch of {len(valid_texts_in_batch)} texts! Mismatch. Batch starting index: {start_index}")
                             # 尝试基于 index 匹配，如果存在的话
                             temp_results = {}
                             for item in api_embeddings:
                                 if isinstance(item, dict) and "index" in item and "embedding" in item:
                                     api_idx = item["index"]
                                     if 0 <= api_idx < len(valid_texts_in_batch):
                                         original_batch_idx = valid_original_indices[api_idx]
                                         temp_results[original_batch_idx] = item["embedding"]
                                     else:
                                         logger.warning(f"API returned embedding with out-of-bounds index {api_idx}. Batch size: {len(valid_texts_in_batch)}")
                             if len(temp_results) > 0:
                                  logger.warning("Partial recovery based on indices.")
                                  batch_results = temp_results
                             else: batch_results = None # 无法恢复，标记失败
                        else: # 数量匹配，按顺序填充
                            for api_idx, item in enumerate(api_embeddings):
                                if isinstance(item, dict) and "embedding" in item and isinstance(item["embedding"], list):
                                    original_batch_idx = valid_original_indices[api_idx] # 获取对应的原始批次索引
                                    batch_results[original_batch_idx] = item["embedding"]
                                else:
                                    logger.error(f"Invalid embedding format in response for item {api_idx} in batch starting at {start_index}. Item: {item}")
                        break # 成功处理，跳出重试循环
                    else:
                        logger.error(f"Unexpected Embedding API response structure for batch starting at {start_index}: {response_data}")
                        break # 结构错误，不重试
                except httpx.TimeoutException:
                    logger.warning(f"Embedding batch request timed out (attempt {retry_count + 1}/{max_retries + 1}). Batch start index: {start_index}")
                    retry_count += 1; await asyncio.sleep(wait_time); wait_time *= 2
                except httpx.HTTPStatusError as status_err:
                    if status_err.response.status_code >= 500 or status_err.response.status_code == 429:
                        logger.warning(f"HTTP error {status_err.response.status_code} getting embedding batch (attempt {retry_count + 1}/{max_retries + 1}). Batch start index: {start_index}. Retrying...")
                        retry_count += 1; await asyncio.sleep(wait_time); wait_time *= 2
                    else:
                        logger.error(f"Non-retryable HTTP error getting embedding batch: {status_err.response.status_code} - {status_err.response.text[:200]}. Batch start index: {start_index}", exc_info=False)
                        break # 非 429/5xx 错误，不重试
                except httpx.RequestError as req_err:
                    logger.error(f"httpx request error getting embedding batch: {req_err}. Batch start index: {start_index}", exc_info=True)
                    retry_count += 1; await asyncio.sleep(wait_time); wait_time *= 2
                except Exception as e:
                    logger.error(f"Unexpected error getting embedding batch starting at {start_index}: {e}", exc_info=True)
                    break # 未知错误，不重试
            # 将批次结果填充回主列表 all_embeddings
            if batch_results is not None:
                for original_batch_idx, embedding in batch_results.items():
                    final_index = start_index + original_batch_idx
                    if 0 <= final_index < len(all_embeddings):
                        all_embeddings[final_index] = embedding
                    else:
                         logger.error(f"Calculated final index {final_index} is out of bounds for all_embeddings (size {len(all_embeddings)}).")
            else:
                 logger.error(f"Failed to get embeddings for batch starting at index {start_index} after all retries.")
                 # all_embeddings 中对应的位置将保持为 None
        tasks_to_process.append(asyncio.create_task(process_single_batch(batch_texts, i)))
    # 并发执行所有批次的处理任务
    if tasks_to_process:
        await asyncio.gather(*tasks_to_process)
    logger.info(f"Finished processing all {len(tasks_to_process)} embedding batches for {len(texts)} texts.")
    return all_embeddings


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算两个向量之间的余弦相似度"""
    if not vec1 or not vec2:
        return 0.0
    vec1_arr = np.array(vec1)
    vec2_arr = np.array(vec2)

    dot_product = np.dot(vec1_arr, vec2_arr)
    norm_vec1 = np.linalg.norm(vec1_arr)
    norm_vec2 = np.linalg.norm(vec2_arr)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0 # 避免除以零
    similarity = dot_product / (norm_vec1 * norm_vec2)
    # 将相似度限制在 [0, 1] 范围内（理论上是[-1, 1]，但 embedding 通常非负）
    return max(0.0, min(1.0, similarity))





async def _get_single_embedding(
    text: str,
    api_key: str,
    api_url: str,
    model: str,
    client: httpx.AsyncClient # 接收共享的 client
) -> Optional[List[float]]:
    """[内部] 使用 API 获取单段文本的嵌入 (需要传入 httpx client)"""
    # ... (保持不变) ...
    if not text or not text.strip():
        logger.warning("Skipping embedding for empty or whitespace-only text.")
        return None
    if len(text) > MAX_CHARS_FOR_EMBEDDING_API_LIMIT:
        logger.warning(f"Chunk text length ({len(text)}) exceeds assumed API limit ({MAX_CHARS_FOR_EMBEDDING_API_LIMIT}). Truncating. Text start: {text[:100]}...")
        text = text[:MAX_CHARS_FOR_EMBEDDING_API_LIMIT]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"input": text, "model": model}
    retry_count = 0; max_retries = 2; wait_time = 1.0
    while retry_count <= max_retries:
        try:
            response = await client.post(api_url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            response_data = response.json()
            if "data" in response_data and isinstance(response_data["data"], list) and response_data["data"]:
                embedding_obj = response_data["data"][0]
                if "embedding" in embedding_obj and isinstance(embedding_obj["embedding"], list):
                     embedding_vector = embedding_obj["embedding"]
                     if all(isinstance(n, (int, float)) for n in embedding_vector):
                         return embedding_vector
                     else: logger.error(f"Embedding API returned invalid vector type: {type(embedding_vector)}"); return None
                else: logger.error(f"Unexpected embedding object structure in API response: {embedding_obj}"); return None
            else: logger.error(f"Unexpected Embedding API response structure: {response_data}"); return None
        except httpx.TimeoutException:
             logger.warning(f"Embedding request timed out (attempt {retry_count + 1}/{max_retries + 1}).")
             retry_count += 1; await asyncio.sleep(wait_time); wait_time *= 2
        except httpx.HTTPStatusError as status_err:
            if status_err.response.status_code >= 500 or status_err.response.status_code == 429:
                logger.warning(f"HTTP error {status_err.response.status_code} getting embedding (attempt {retry_count + 1}/{max_retries + 1}): {status_err.response.text[:200]}. Retrying...")
                retry_count += 1; await asyncio.sleep(wait_time); wait_time *= 2
            else:
                logger.error(f"Non-retryable HTTP error getting single embedding: {status_err.response.status_code} - {status_err.response.text[:200]}", exc_info=False)
                return None
        except httpx.RequestError as req_err:
             logger.error(f"httpx request error getting embedding: {req_err}", exc_info=True)
             retry_count += 1; await asyncio.sleep(wait_time); wait_time *= 2
        except Exception as e:
             logger.error(f"Unexpected error getting single embedding: {e}", exc_info=True)
             return None
    logger.error(f"Failed to get embedding after {max_retries + 1} attempts.")
    return None
# get_embedding_dimension 保持不变
async def get_embedding_dimension(
    api_key: str,
    api_url: str,
    model_name: str,
    client: httpx.AsyncClient
) -> Optional[int]:
    """通过嵌入一个测试字符串来动态获取模型的输出维度。"""
    # ... (保持不变) ...
    test_string = "dimension check"
    logger.info(f"Performing a test embedding with model '{model_name}' to determine dimension...")
    try:
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
# --- [重构] 下载/保存函数 ---
# --- [重构] 下载/保存函数 ---
async def download_content_file(paper_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取单篇论文内容文件（PMC EuropePMC Direct PDF -> SciHub PDF -> Abstract MD）。
    仅下载或保存文件，不处理内容。
    # ... (函数文档和参数不变) ...
    """
    pmid = paper_input["pmid"]
    pmcid = paper_input.get("pmcid")
    doi = paper_input.get("doi")
    abstract = paper_input.get("abstract")
    title = paper_input.get("title", "N/A")
    db_name = paper_input["db_name"]
    http_client = paper_input["http_client"]
    storage_path = os.path.join(CHROMA_DB_PATH, db_name)
    os.makedirs(storage_path, exist_ok=True)
    pdf_file_path = os.path.join(storage_path, f"{pmid}.pdf")
    md_file_path = os.path.join(storage_path, f"{pmid}.md")
    final_file_path: Optional[str] = None
    is_pdf_file: bool = False
    # pdf_url: Optional[str] = None # 我们将直接尝试下载，而不是先获取URL再下载
    final_source: str = "N/A"
    success: bool = False

    # --- 1. 尝试从 EuropePMC 直接下载 PDF ---
    if pmcid:
        # 清理 PMCID，EuropePMC 通常需要纯数字的 accid
        cleaned_pmcid = pmcid.replace("PMC", "") if isinstance(pmcid, str) else None
        if cleaned_pmcid and cleaned_pmcid.isdigit():
            europe_pmc_direct_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
            current_source_attempt = "EuropePMC Direct PDF"
            logger.info(f"PMID {pmid}: Attempting to download PDF directly from EuropePMC: {europe_pmc_direct_url}")

            if await download_file(europe_pmc_direct_url, pdf_file_path, http_client):
                final_file_path = pdf_file_path
                is_pdf_file = True
                final_source = f"{current_source_attempt} File"
                success = True
                logger.info(f"PMID {pmid}: Successfully downloaded PDF from EuropePMC to {pdf_file_path}")
                # 清理可能存在的旧 .md 文件
                if os.path.exists(md_file_path):
                    try: await asyncio.to_thread(os.remove, md_file_path); logger.debug(f"Removed potentially conflicting MD file: {md_file_path}")
                    except OSError as e_rem: logger.warning(f"Could not remove potentially conflicting MD file {md_file_path}: {e_rem}")
            else:
                logger.warning(f"PMID {pmid}: Failed to download PDF from EuropePMC: {europe_pmc_direct_url}")
                final_source = f"{current_source_attempt} Download Failed"
                # success 保持 False，将继续尝试 Sci-Hub
        else:
            logger.warning(f"PMID {pmid}: PMCID '{pmcid}' is not in a valid format for EuropePMC direct link. Skipping this method.")
            final_source = "EuropePMC Direct Link Skipped (Invalid PMCID Format)"
            # success 保持 False
    else:
        logger.info(f"PMID {pmid}: No PMCID provided, skipping EuropePMC direct PDF attempt.")
        final_source = "No PMCID for EuropePMC" # success 保持 False

    # --- 2. 如果 EuropePMC 下载失败 或 未尝试，并且有 DOI，则尝试 Sci-Hub ---
    if not success and doi: # 只有在上面没有成功下载PDF，并且有DOI时才尝试Sci-Hub
        # 之前的 final_source 记录了 EuropePMC 尝试的结果，这里可以据此调整日志
        logger.info(f"PMID {pmid}: Previous PDF attempt ({final_source}) was unsuccessful. Trying Sci-Hub via DOI: {doi}")
        sci_hub_pdf_url = await asyncio.to_thread(run_scihub_web_request, doi) # run_scihub_web_request 返回的是URL
        if sci_hub_pdf_url:
            current_source_attempt = "Sci-Hub PDF"
            logger.info(f"PMID {pmid}: Found PDF link via Sci-Hub: {sci_hub_pdf_url}. Attempting download.")
            if await download_file(sci_hub_pdf_url, pdf_file_path, http_client):
                final_file_path = pdf_file_path
                is_pdf_file = True
                final_source = f"{current_source_attempt} File" # 覆盖之前的失败源
                success = True
                logger.info(f"PMID {pmid}: Successfully downloaded PDF from Sci-Hub to {pdf_file_path}")
                # 清理可能存在的旧 .md 文件
                if os.path.exists(md_file_path):
                    try: await asyncio.to_thread(os.remove, md_file_path); logger.debug(f"Removed potentially conflicting MD file: {md_file_path}")
                    except OSError as e_rem: logger.warning(f"Could not remove potentially conflicting MD file {md_file_path}: {e_rem}")
            else:
                logger.warning(f"PMID {pmid}: Found Sci-Hub link but failed to download PDF: {sci_hub_pdf_url}")
                # final_source 可能是 EuropePMC 失败的状态，或者更新为 Sci-Hub 下载失败
                final_source = f"{current_source_attempt} Download Failed" # 明确是 Sci-Hub 下载失败
                # success 保持 False
        else:
            logger.info(f"PMID {pmid}: Could not find PDF link via Sci-Hub. DOI: {doi}")
            # 更新 final_source 以反映 Sci-Hub 链接获取失败
            final_source = "Sci-Hub Link Not Found" # 明确是 Sci-Hub 链接未找到
            # success 保持 False

    # --- 3. 如果没有下载 PDF，使用摘要并保存为 MD --- (原步骤4)
    if not success: # 仅在 PDF 获取不成功时执行
        logger.info(f"PMID {pmid}: No PDF obtained ({final_source}). Falling back to abstract.")
        if abstract and abstract.strip():
            try:
                # 使用异步线程写入文件
                async with asyncio.Lock():
                    with open(md_file_path, 'w', encoding='utf-8') as f:
                        f.write(f"# {title}\n\n")
                        f.write(f"PMID: {pmid}\n")
                        if doi: f.write(f"DOI: {doi}\n")
                        if pmcid: f.write(f"PMCID: {pmcid}\n")
                        f.write("\n---\n\n")
                        f.write(abstract)
                final_file_path = md_file_path
                is_pdf_file = False
                final_source = "Abstract (MD File)" # 覆盖之前的失败源
                success = True
                logger.info(f"PMID {pmid}: Saved abstract to {md_file_path}")
                # 清理可能存在的旧 .pdf 文件
                if os.path.exists(pdf_file_path):
                    try: await asyncio.to_thread(os.remove, pdf_file_path); logger.debug(f"Removed potentially conflicting PDF file: {pdf_file_path}")
                    except OSError as e_rem: logger.warning(f"Could not remove potentially conflicting PDF file {pdf_file_path}: {e_rem}")
            except IOError as e:
                logger.error(f"PMID {pmid}: Failed to save abstract to MD file {md_file_path}: {e}", exc_info=True)
                final_source = "Abstract Save Failed" # 覆盖之前的失败源
                success = False
        else:
            logger.warning(f"PMID {pmid}: No PDF obtained and abstract is empty or missing. Cannot process.")
            # final_source 此时应该已经是之前某一步的失败状态
            if not (final_source.endswith("Failed") or \
                    final_source.endswith("Skipped (Invalid PMCID Format)") or \
                    final_source == "No PMCID for EuropePMC" or \
                    final_source.endswith("Not Found")):
                 final_source = "No Content Available"
            success = False

    return {
        "pmid": pmid,
        "title": title,
        "doi": doi,
        "pmcid": pmcid,
        "file_path": final_file_path,
        "is_pdf": is_pdf_file,
        "source": final_source,
        "success": success
    }

# --- [新增] MD 文件处理与嵌入函数 ---
async def process_md_file_and_embed(
    md_file_path: str,
    metadata: Dict[str, Any],
    http_client: httpx.AsyncClient,
    expected_dimension: int
) -> Optional[List[Dict]]:
    """
    读取 MD 文件内容，进行语义分块（内部已使用批量嵌入），然后对最终块进行批量嵌入。
    """
    pmid = metadata.get("pmid", os.path.basename(md_file_path).replace(".md", ""))
    final_source = metadata.get("source", "Unknown MD")
    logger.info(f"PMID {pmid}: Processing MD file for embedding: {md_file_path} (Source: {final_source})")
    text_to_process: Optional[str] = None
    try:
        # ... (文件读取逻辑保持不变) ...
        async with asyncio.Lock():
             with open(md_file_path, 'r', encoding='utf-8') as f:
                 text_to_process = f.read()
        if not text_to_process or not text_to_process.strip():
            logger.warning(f"PMID {pmid}: MD file {md_file_path} is empty.")
            return None
        logger.info(f"PMID {pmid}: Successfully read text (approx {len(text_to_process)} chars) from MD file.")
    except Exception as e:
        logger.error(f"PMID {pmid}: Error reading MD file {md_file_path}: {e}", exc_info=True)
        return None
    # --- 语义分块 (内部已使用批量嵌入) ---
    logger.info(f"PMID {pmid}: Applying paragraph-based splitting to the text.")
    embedding_config_for_split = {
        "api_key": SILICONFLOW_API_KEY,
        "api_url": SILICONFLOW_API_URL,
        "model": SILICONFLOW_EMBEDDING_MODEL,
        "pmid": pmid
    }
    try:
        # 调用新的文本分割函数
        chunks = paragraph_based_split_text(
            text=text_to_process,
            max_chunk_chars=15000, # 要求的最大字符数
            min_chunk_chars=100    # 要求的最小字符数
        )
    except Exception as split_err:
         # 如果 paragraph_based_split_text 内部发生意外错误
         logger.error(f"PMID {pmid}: Paragraph-based splitting failed critically with error: {split_err}", exc_info=True)
         logger.warning(f"PMID {pmid}: Falling back to character-based splitting.")
         # 使用旧的 CONTENT_CHUNK_SIZE 和 CONTENT_CHUNK_OVERLAP 作为后备方案的参数
         chunks = split_text_into_chunks(text_to_process, CONTENT_CHUNK_SIZE, CONTENT_CHUNK_OVERLAP)
    
    if not chunks:
        logger.warning(f"PMID {pmid}: Text splitting (paragraph-based or fallback) resulted in zero chunks from MD file.")
        return None
    logger.info(f"PMID {pmid}: Split MD content into {len(chunks)} final chunks for embedding using paragraph-based method (or fallback).")
    # --- 修改点：对最终块进行批量嵌入 ---
    logger.info(f"PMID {pmid}: Starting batch embedding for {len(chunks)} final chunks...")
    try:
        chunk_embeddings_results = await _get_embeddings_batch(
            chunks,
            SILICONFLOW_API_KEY,
            SILICONFLOW_API_URL,
            SILICONFLOW_EMBEDDING_MODEL,
            http_client
        )
    except Exception as final_batch_err:
        logger.error(f"PMID {pmid}: Error during batch embedding for final chunks: {final_batch_err}", exc_info=True)
        return None # 无法嵌入最终块，返回失败
    logger.info(f"PMID {pmid}: Finished batch embedding tasks for final chunks.")
    # --- 结束修改点 ---
    # --- 整理结果 (逻辑不变，但现在处理的是批量结果) ---
    chunks_data_list = []
    successful_embeddings_count = 0
    failed_chunk_count = 0
    for i, embedding_result in enumerate(chunk_embeddings_results):
        chunk_text = chunks[i]
        chunk_id = f"{pmid}_chunk_{i}"
        if embedding_result is not None and isinstance(embedding_result, list):
            if len(embedding_result) != expected_dimension:
                 logger.error(f"PMID {pmid} Final Chunk {i}: Embedding dimension mismatch! Got {len(embedding_result)}, expected {expected_dimension}. Skipping.")
                 failed_chunk_count += 1
                 continue
            # 构建元数据
            chunk_metadata = {
                "title": metadata.get("title", "N/A"),
                "doi": metadata.get("doi"),
                "pmid": pmid,
                "pmcid": metadata.get("pmcid"),
                "chunk_index": i,
                "source": final_source,
                "split_method": "semantic" # 或者根据实际情况标记 fallback
            }
            chunk_metadata = {k: v for k, v in chunk_metadata.items() if v is not None}
            chunks_data_list.append({
                "id": chunk_id,
                "embedding": embedding_result,
                "metadata": chunk_metadata,
                "document": chunk_text
            })
            successful_embeddings_count += 1
        else:
            logger.warning(f"PMID {pmid} Final Chunk {i}: Failed to get embedding from batch result. Skipping.")
            failed_chunk_count += 1
    if failed_chunk_count > 0:
         logger.warning(f"PMID {pmid}: Failed to embed {failed_chunk_count} out of {len(chunks)} final chunks.")
    if not chunks_data_list:
        logger.warning(f"PMID {pmid}: Failed to embed any final chunks successfully from MD file: {md_file_path}")
        return None
    logger.info(f"PMID {pmid}: Successfully embedded {successful_embeddings_count} out of {len(chunks)} final chunks from MD file.")
    return chunks_data_list


# --- 主 MCP 工具函数 (search_literature) ---
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
    搜索文献、获取原文或摘要、下载保存为 PDF/MD 文件、使用 marker 将 PDF 转为 MD、
    然后将 MD 文件内容分块嵌入并存储到向量数据库。同时，将所有文献的PMID及其摘要合并存储为一个单独条目。
    Args:
        keyword (str): PubMed 搜索关键词。
        db_name (str): ChromaDB 集合名称，也用作本地文件存储的子目录名。
        min_date (Optional[str]): 检索的起始出版日期 (格式: YYYY/MM/DD 或 YYYY)。默认为 None。
        max_date (Optional[str]): 检索的结束出版日期 (格式: YYYY/MM/DD 或 YYYY)。默认为 None。
        sort_by (str): 结果排序方式。可选: "relevance", "pub_date"。默认为 "relevance"。
        num_results (int): 希望检索的最大文献数量。默认为 DEFAULT_RETMAX_PUBMED。
    Returns:
        Dict[str, Any]: 操作结果摘要。
            {
                "success": bool,
                "message": str,
                "papers_found": int,        # PubMed 找到的有 PMID 的文献数
                "download_tasks_run": int,  # 运行的下载/保存任务数
                "files_saved_count": int,   # 成功保存本地文件(PDF/MD)的数量
                "pdfs_moved_for_marker": int, # 移动到待处理文件夹的 PDF 数量
                "marker_processed_success": int, # marker 成功转换并移回的 MD 文件数
                "marker_processed_failure": int, # marker 处理失败的 PDF 数 (或未生成MD)
                "embedding_tasks_run": int, # 运行的 MD 嵌入任务数
                "papers_embedded": int,     # 成功生成嵌入块的论文数
                "chunks_embedded": int,     # 成功嵌入并存储的文献块总数量
                "pmid_abstract_entry_saved": bool, # 是否成功保存了 PMID+摘要合集条目
                "embedding_dimension": Optional[int], # 确定的维度
                "db_path": str, # ChromaDB 基础路径
                "storage_path": str, # 本地文件存储路径 (子目录)
                "collection_name": str,
                "errors": List[str]
            }
    """
    start_time = time.time()
    logger.info(f"Starting literature search: keyword='{keyword}', db_name='{db_name}', num_results={num_results}, sort='{sort_by}', min_date='{min_date}', max_date='{max_date}'")
    storage_path = os.path.join(CHROMA_DB_PATH, db_name)
    pdf_process_dir = os.path.join(storage_path, "pdf_to_process") # [新增] Marker 处理目录
    results = {
        "success": False, "message": "", "papers_found": 0, "download_tasks_run": 0,
        "files_saved_count": 0, "pdfs_moved_for_marker": 0,
        "marker_processed_success": 0, "marker_processed_failure": 0,
        "embedding_tasks_run": 0, "papers_embedded": 0, "chunks_embedded": 0,
        "pmid_abstract_entry_saved": False, # [修改] 键名
        "title_pmid_doi_entry_saved": False, # [新增] 新的标志
        "embedding_dimension": None,
        "db_path": CHROMA_DB_PATH,
        "storage_path": storage_path,
        "collection_name": db_name, "errors": []
    }
    # 创建共享的 httpx 客户端
    async with httpx.AsyncClient(timeout=180.0, limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)) as http_client:
        try:
            # --- 1. 动态确定 Embedding 维度 ---
            determined_embedding_dimension = await get_embedding_dimension(
                SILICONFLOW_API_KEY, SILICONFLOW_API_URL, SILICONFLOW_EMBEDDING_MODEL, http_client
            )
            if determined_embedding_dimension is None:
                error_msg = f"Failed to determine embedding dimension for model '{SILICONFLOW_EMBEDDING_MODEL}'. Cannot proceed."
                logger.error(error_msg); results["message"] = error_msg; results["errors"].append(error_msg)
                return results
            results["embedding_dimension"] = determined_embedding_dimension
            logger.info(f"Using dynamically determined embedding dimension: {determined_embedding_dimension}")
            # --- 2. 输入参数校验 ---
            if sort_by not in ["relevance", "pub_date"]:
                logger.warning(f"Invalid sort_by value '{sort_by}'. Using default 'relevance'."); sort_by = "relevance"
            if num_results <= 0:
                logger.warning(f"Invalid num_results value {num_results}. Using default {DEFAULT_RETMAX_PUBMED}."); num_results = DEFAULT_RETMAX_PUBMED
            if not re.match(r'^[a-zA-Z0-9_-]+$', db_name):
                 error_msg = f"Invalid db_name '{db_name}'. Must contain only letters, numbers, underscores, or hyphens."
                 logger.error(error_msg); results["message"] = error_msg; results["errors"].append(error_msg)
                 return results
            # --- 3. 搜索 PubMed & 收集摘要与元数据 ---
            pubmed_results = await _search_pubmed_internal(
                keyword, retmax=num_results, sort_by=sort_by, min_date=min_date, max_date=max_date
            )
            raw_papers_found = len(pubmed_results)
            logger.info(f"PubMed search found {raw_papers_found} initial results.")
            papers_to_process_input = []
            pmid_abstract_pairs = [] # [修改] 存储 pmid 和 abstract 对
            papers_with_pmid_count = 0
            for paper in pubmed_results:
                if not paper.get('pmid'):
                     logger.warning(f"Skipping paper due to missing PMID. Title: {paper.get('title', 'N/A')}")
                     results["errors"].append(f"Skipped paper: Missing PMID (Title: {paper.get('title', 'N/A')})")
                     continue
                papers_with_pmid_count += 1
                pmid = paper['pmid']
                abstract = paper.get('abstract')
                # 收集 PMID 和摘要对
                if abstract and isinstance(abstract, str) and abstract.strip():
                    pmid_abstract_pairs.append({"pmid": pmid, "abstract": abstract.strip()})
                # 准备下载任务输入
                paper_download_input = {
                    "pmid": pmid,
                    "pmcid": paper.get('pmcid'),
                    "doi": paper.get('doi'),
                    "title": paper.get('title', 'N/A'),
                    "abstract": abstract, # 仍然传递摘要，以便在 PDF 失败时使用
                    "db_name": db_name,
                    "http_client": http_client,
                }
                papers_to_process_input.append(paper_download_input)
            results["papers_found"] = papers_with_pmid_count
            if not papers_to_process_input:
                 results["success"] = True; results["message"] = f"PubMed search found {raw_papers_found} results, but none had a valid PMID for processing."
                 logger.warning(results["message"])
                 return results
            logger.info(f"Prepared {len(papers_to_process_input)} papers with PMIDs for content download/saving.")
            logger.info(f"Collected {len(pmid_abstract_pairs)} non-empty abstracts for combined storage.")
            # --- 4. 准备 ChromaDB ---
            try:
                chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                logger.info(f"Attempting to get or create collection '{db_name}' with hnsw:space='cosine'.")
                collection = await asyncio.to_thread(
                    chroma_client.get_or_create_collection,
                    name=db_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Successfully got or created ChromaDB collection '{db_name}'.")
                # 可选：维度检查
                # ... (之前的维度检查逻辑可以保留) ...
            except Exception as e:
                error_msg = f"Failed to initialize ChromaDB collection '{db_name}': {e}"
                logger.error(error_msg, exc_info=True); results["message"] = error_msg; results["errors"].append(f"{error_msg} (Traceback logged)")
                return results
            # --- 5. 存储合并 PMID+摘要 (使用占位符 Embedding 和 Metadata) ---
            if pmid_abstract_pairs:
                # [修改] 将 pmid+摘要 列表转为 JSON 字符串存储
                combined_abstract_json = await asyncio.to_thread(json.dumps, pmid_abstract_pairs, ensure_ascii=False, indent=2)
                combined_abstract_id = ALL_ABSTRACTS_DOC_ID
                placeholder_text = "Placeholder text for combined pmid-abstracts entry"
                try:
                    logger.info(f"Generating placeholder embedding for combined PMID-abstracts entry (ID: {combined_abstract_id}).")
                    placeholder_embedding = await _get_single_embedding(
                        placeholder_text, SILICONFLOW_API_KEY, SILICONFLOW_API_URL, SILICONFLOW_EMBEDDING_MODEL, http_client
                    )
                    if placeholder_embedding and len(placeholder_embedding) == determined_embedding_dimension:
                        combined_abstract_metadata = {COMBINED_ABSTRACT_METADATA_KEY: combined_abstract_json} # 使用新 key 和 json
                        logger.info(f"Attempting to save/update combined PMID-abstracts entry (ID: {combined_abstract_id}).")
                        await asyncio.to_thread(
                            collection.upsert,
                            ids=[combined_abstract_id],
                            embeddings=[placeholder_embedding],
                            metadatas=[combined_abstract_metadata],
                            documents=[placeholder_text]
                        )
                        results["pmid_abstract_entry_saved"] = True # [修改] 键名
                        logger.info(f"Successfully saved/updated combined PMID-abstracts entry (ID: {combined_abstract_id}).")
                    elif placeholder_embedding:
                         error_msg = f"Placeholder embedding dimension mismatch. Cannot save combined PMID-abstracts."
                         logger.error(error_msg); results["errors"].append(error_msg); results["pmid_abstract_entry_saved"] = False
                    else:
                        error_msg = f"Failed to generate placeholder embedding for combined PMID-abstracts. Cannot save."
                        logger.error(error_msg); results["errors"].append(error_msg); results["pmid_abstract_entry_saved"] = False
                except Exception as e:
                    error_msg = f"Failed to save combined PMID-abstracts entry: {e}"; logger.error(error_msg, exc_info=True)
                    results["errors"].append(f"{error_msg} (Traceback logged)"); results["pmid_abstract_entry_saved"] = False
            else:
                logger.warning("No PMID-abstract pairs found to combine and save.")
                # 可选：删除旧条目
                # ... (之前的删除逻辑可以保留) ...
            # --- [新增逻辑开始] 5.1 存储所有文献的 Title, PMID, DOI ---
            TITLE_PMID_DOI_DOC_ID = "title_pmid_doi" # 新增的文档 ID
            TITLE_PMID_DOI_METADATA_KEY = "title_pmid_doi_json" # 新增的 metadata key
            title_pmid_doi_list = []
            # 从已准备好的 papers_to_process_input 中提取信息更方便
            for paper_input in papers_to_process_input:
                 entry = {
                     "title": paper_input.get("title", "N/A"),
                     "pmid": paper_input.get("pmid"), # 必须有 pmid
                     "doi": paper_input.get("doi", None) # doi 可能为 None
                 }
                 # 确保 PMID 存在才添加
                 if entry["pmid"]:
                     title_pmid_doi_list.append(entry)
            results["title_pmid_doi_entry_saved"] = False # 初始化新标志
            if title_pmid_doi_list:
                try:
                    # 将列表转为 JSON 字符串存储
                    title_pmid_doi_json = await asyncio.to_thread(json.dumps, title_pmid_doi_list, ensure_ascii=False, indent=2)
                    placeholder_text_tpd = f"Placeholder for {len(title_pmid_doi_list)} title/pmid/doi entries"
                    logger.info(f"Generating placeholder embedding for title/pmid/doi list (ID: {TITLE_PMID_DOI_DOC_ID}).")
                    placeholder_embedding_tpd = await _get_single_embedding(
                        placeholder_text_tpd, SILICONFLOW_API_KEY, SILICONFLOW_API_URL, SILICONFLOW_EMBEDDING_MODEL, http_client
                    )
                    if placeholder_embedding_tpd and len(placeholder_embedding_tpd) == determined_embedding_dimension:
                        metadata_tpd = {TITLE_PMID_DOI_METADATA_KEY: title_pmid_doi_json}
                        logger.info(f"Attempting to save/update title/pmid/doi list entry (ID: {TITLE_PMID_DOI_DOC_ID}).")
                        await asyncio.to_thread(
                            collection.upsert,
                            ids=[TITLE_PMID_DOI_DOC_ID],
                            embeddings=[placeholder_embedding_tpd],
                            metadatas=[metadata_tpd],
                            documents=[placeholder_text_tpd] # 存储占位符文本
                        )
                        results["title_pmid_doi_entry_saved"] = True
                        logger.info(f"Successfully saved/updated title/pmid/doi list entry (ID: {TITLE_PMID_DOI_DOC_ID}).")
                    elif placeholder_embedding_tpd:
                        error_msg = f"Placeholder embedding dimension mismatch for title/pmid/doi list. Cannot save."
                        logger.error(error_msg); results["errors"].append(error_msg)
                    else:
                        error_msg = f"Failed to generate placeholder embedding for title/pmid/doi list. Cannot save."
                        logger.error(error_msg); results["errors"].append(error_msg)
                except Exception as e:
                    error_msg = f"Failed to save title/pmid/doi list entry: {e}"; logger.error(error_msg, exc_info=True)
                    results["errors"].append(f"{error_msg} (Traceback logged)")
            else:
                logger.warning("No valid title/pmid/doi entries found to save.")
                # 可选：如果列表为空，尝试删除旧条目（如果需要）
                try:
                    logger.info(f"Attempting to delete potentially obsolete title/pmid/doi entry (ID: {TITLE_PMID_DOI_DOC_ID}) as no new data found.")
                    await asyncio.to_thread(collection.delete, ids=[TITLE_PMID_DOI_DOC_ID])
                    logger.info(f"Successfully deleted potentially obsolete title/pmid/doi entry.")
                except Exception as delete_err:
                     # 通常忽略删除错误，因为它可能本身就不存在
                     logger.debug(f"Could not delete title/pmid/doi entry (may not exist): {delete_err}")

            # --- 6. 并发下载/保存原文或摘要 ---
            download_tasks = []
            for paper_input in papers_to_process_input:
                download_tasks.append(asyncio.create_task(download_content_file(paper_input)))
            logger.info(f"Starting {len(download_tasks)} download/save tasks...")
            download_results = await asyncio.gather(*download_tasks, return_exceptions=True)
            logger.info("Finished download/save tasks.")
            results["download_tasks_run"] = len(download_results)
            successful_saves = [] # 存储成功保存的文件信息 (包含元数据)
            files_saved_count = 0
            pdfs_to_move = [] # 存储需要移动的 PDF 文件路径和 PMID
            for i, result_or_exc in enumerate(download_results):
                pmid_for_log = papers_to_process_input[i].get("pmid", f"task_{i}")
                if isinstance(result_or_exc, Exception):
                    error_msg = f"Error downloading/saving content for PMID {pmid_for_log}: {type(result_or_exc).__name__} - {result_or_exc}"
                    logger.error(f"Exception in download task for PMID {pmid_for_log}:", exc_info=result_or_exc)
                    results["errors"].append(f"{error_msg} (Traceback logged)")
                elif result_or_exc is None:
                    logger.warning(f"Download task for PMID {pmid_for_log} returned None.")
                    results["errors"].append(f"Download task failed unexpectedly for PMID {pmid_for_log}")
                elif isinstance(result_or_exc, dict):
                    if result_or_exc.get("success"):
                        files_saved_count += 1
                        successful_saves.append(result_or_exc) # 保存整个结果字典
                        if result_or_exc.get("is_pdf") and result_or_exc.get("file_path"):
                            pdfs_to_move.append({
                                "src_path": result_or_exc["file_path"],
                                "pmid": result_or_exc["pmid"]
                            })
                        logger.info(f"Successfully saved content for PMID {pmid_for_log}. Source: {result_or_exc.get('source')}, Path: {result_or_exc.get('file_path')}")
                    else:
                        logger.warning(f"Failed to save content for PMID {pmid_for_log}. Source: {result_or_exc.get('source')}")
                        results["errors"].append(f"Failed to save content for PMID {pmid_for_log} (Source: {result_or_exc.get('source')})")
                else:
                    logger.error(f"Unexpected result type '{type(result_or_exc)}' from download task for PMID {pmid_for_log}.")
                    results["errors"].append(f"Unexpected result type for PMID {pmid_for_log}: {type(result_or_exc).__name__}")
            results["files_saved_count"] = files_saved_count
            logger.info(f"Total files saved: {files_saved_count}. PDFs needing marker processing: {len(pdfs_to_move)}")
            # --- 7. 移动 PDF 并执行 Marker ---
            pdfs_moved_count = 0
            marker_success_count = 0
            marker_failure_count = 0
            marker_processed_pmids = set() # 记录哪些 pmid 的 PDF 被 marker 处理了
            if pdfs_to_move:
                try:
                    # 创建 pdf_to_process 目录 (使用异步线程以防阻塞)
                    await asyncio.to_thread(os.makedirs, pdf_process_dir, exist_ok=True)
                    logger.info(f"Created marker processing directory: {pdf_process_dir}")
                    # 移动 PDF 文件
                    move_tasks = []
                    pmids_being_moved = []
                    for pdf_info in pdfs_to_move:
                        src = pdf_info["src_path"]
                        pmid = pdf_info["pmid"]
                        dest = os.path.join(pdf_process_dir, f"{pmid}.pdf")
                        move_tasks.append(asyncio.to_thread(shutil.copy2, src, dest))
                        pmids_being_moved.append(pmid)
                    logger.info(f"Moving {len(move_tasks)} PDF files to {pdf_process_dir}...")
                    move_results = await asyncio.gather(*move_tasks, return_exceptions=True)
                    for i, res in enumerate(move_results):
                        pmid_moved = pmids_being_moved[i]
                        if isinstance(res, Exception):
                            logger.error(f"Failed to move PDF for PMID {pmid_moved}: {res}")
                            results["errors"].append(f"Failed to move PDF for marker processing (PMID: {pmid_moved}): {res}")
                        else:
                            pdfs_moved_count += 1
                            logger.debug(f"Successfully moved PDF for PMID {pmid_moved} to processing directory.")
                    results["pdfs_moved_for_marker"] = pdfs_moved_count
                    logger.info(f"Successfully moved {pdfs_moved_count} PDFs for marker processing.")
                except Exception as move_prep_err:
                    logger.error(f"Error preparing or moving PDFs for marker: {move_prep_err}", exc_info=True)
                    results["errors"].append(f"Error during PDF move stage: {move_prep_err}")
                    # 如果移动阶段出错，跳过 marker
                    pdfs_moved_count = 0 # 重置计数，因为无法继续
                # 只有成功移动了 PDF 才执行 marker
                if pdfs_moved_count > 0:
                    marker_cmd = f"marker \"{pdf_process_dir}\" --workers {MARKER_WORKERS} --output_dir \"{pdf_process_dir}\" --output_format markdown"
                    logger.info(f"Executing marker command: {marker_cmd}")
                    try:
                        # 使用 asyncio.create_subprocess_shell 执行 marker
                        process = await asyncio.create_subprocess_shell(
                            marker_cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        stdout, stderr = await process.communicate() # 等待命令完成并获取输出
                        if process.returncode == 0:
                            logger.info("Marker command executed successfully.")
                            if stdout: logger.debug(f"Marker stdout:\n{stdout.decode(errors='ignore')}")
                            if stderr: logger.warning(f"Marker stderr:\n{stderr.decode(errors='ignore')}") # Marker 可能在 stderr 输出进度信息
                            # 处理 marker 输出：移动 MD 文件
                            move_back_tasks = []
                            pmids_to_move_back = []
                            for pmid in pmids_being_moved: # 仅处理那些成功移动过去的PDF对应的PMID
                                marker_output_dir = os.path.join(pdf_process_dir, pmid)
                                marker_md_path = os.path.join(marker_output_dir, f"{pmid}.md")
                                final_md_path = os.path.join(storage_path, f"{pmid}.md")
                                if os.path.exists(marker_md_path):
                                    logger.debug(f"Found marker output MD for PMID {pmid}: {marker_md_path}")
                                    # 使用 to_thread 移动文件并删除目录
                                    async def move_and_cleanup(src_md, dest_md, temp_dir):
                                        try:
                                            shutil.move(src_md, dest_md)
                                            shutil.rmtree(temp_dir) # 删除 pdf_to_process/{pmid} 目录
                                            return True
                                        except Exception as e:
                                            logger.error(f"Error moving/cleaning up marker output for PMID {pmid}: {e}")
                                            return False
                                    move_back_tasks.append(asyncio.create_task(move_and_cleanup(marker_md_path, final_md_path, marker_output_dir)))
                                    pmids_to_move_back.append(pmid)
                                else:
                                    logger.warning(f"Marker command finished, but expected output MD not found for PMID {pmid} at {marker_md_path}")
                                    marker_failure_count += 1
                                    results["errors"].append(f"Marker processed PDF but output MD not found for PMID {pmid}")
                                    # 不需要添加到 marker_processed_pmids
                            if move_back_tasks:
                                logger.info(f"Moving {len(move_back_tasks)} generated MD files back...")
                                move_back_results = await asyncio.gather(*move_back_tasks) # 不需要 return_exceptions=True，因为函数内部处理了
                                for i, success_flag in enumerate(move_back_results):
                                    moved_back_pmid = pmids_to_move_back[i]
                                    if success_flag:
                                        marker_success_count += 1
                                        marker_processed_pmids.add(moved_back_pmid) # 记录成功处理的 PMID
                                        logger.debug(f"Successfully moved back MD for PMID {moved_back_pmid}.")
                                    else:
                                        marker_failure_count += 1
                                        results["errors"].append(f"Failed to move back/cleanup marker MD for PMID {moved_back_pmid}")
                        else:
                            error_msg = f"Marker command failed with return code {process.returncode}."
                            logger.error(error_msg)
                            stderr_output = stderr.decode(errors='ignore') if stderr else "No stderr output."
                            logger.error(f"Marker stderr:\n{stderr_output}")
                            results["errors"].append(f"{error_msg} (PMIDs attempted: {pmids_being_moved}). Stderr logged.")
                            marker_failure_count += pdfs_moved_count # 标记所有移动的 PDF 都处理失败了
                    except FileNotFoundError:
                        error_msg = "FATAL: 'marker' command not found. Please ensure marker-pdf is installed and in the system PATH."
                        logger.error(error_msg)
                        results["message"] = error_msg; results["errors"].append(error_msg)
                        # 这是严重错误，可能无法继续处理PDF，但可以继续处理摘要MD
                        marker_failure_count += pdfs_moved_count
                    except Exception as marker_err:
                        error_msg = f"Error executing marker command: {marker_err}"
                        logger.error(error_msg, exc_info=True)
                        results["errors"].append(f"{error_msg} (Traceback logged)")
                        marker_failure_count += pdfs_moved_count
                    results["marker_processed_success"] = marker_success_count
                    results["marker_processed_failure"] = marker_failure_count
                    logger.info(f"Marker processing finished. Success: {marker_success_count}, Failure/Not Found: {marker_failure_count}")
                    # [可选] 清理 pdf_to_process 目录中未成功处理的文件/目录
                    try:
                        if os.path.exists(pdf_process_dir):
                            logger.info(f"Cleaning up marker processing directory: {pdf_process_dir}")
                            await asyncio.to_thread(shutil.rmtree, pdf_process_dir, ignore_errors=True)
                    except Exception as cleanup_err:
                        logger.warning(f"Error during marker directory cleanup: {cleanup_err}")
            else:
                logger.info("No PDFs found to process with marker.")
            # --- 8. 并发处理 MD 文件（来自 Marker 或 摘要）并嵌入 ---
            embedding_tasks = []
            md_files_to_process_info = {} # 存储 MD 文件路径和对应的元数据
            # 收集所有最终应该存在的 MD 文件及其元数据
            for save_result in successful_saves:
                pmid = save_result["pmid"]
                is_pdf = save_result["is_pdf"]
                final_md_path = os.path.join(storage_path, f"{pmid}.md")
                # 如果原始是 PDF 且 marker 成功处理了它
                if is_pdf and pmid in marker_processed_pmids:
                    if os.path.exists(final_md_path):
                        # 使用 marker 后的来源信息
                        md_files_to_process_info[pmid] = {
                            "md_path": final_md_path,
                            "metadata": {**save_result, "source": "Marker PDF Conversion"} # 更新 source
                        }
                    else: # Marker 声称成功，但文件不在？记录错误
                         logger.error(f"Inconsistency: Marker reported success for PMID {pmid}, but final MD file missing: {final_md_path}")
                         results["errors"].append(f"Marker output MD missing after reported success for PMID {pmid}")
                # 如果原始是 MD (摘要)
                elif not is_pdf and save_result.get("file_path") and save_result["file_path"].endswith(".md"):
                    md_path_from_save = save_result["file_path"]
                    if os.path.exists(md_path_from_save):
                         md_files_to_process_info[pmid] = {
                             "md_path": md_path_from_save,
                             "metadata": save_result # 使用原始保存结果中的元数据
                         }
                    else:
                         logger.warning(f"Abstract MD file was reported saved but not found for PMID {pmid} at {md_path_from_save}")
                         results["errors"].append(f"Saved abstract MD file not found for PMID {pmid}")
                # 其他情况（如PDF下载成功但marker失败/跳过）则不处理
            # 为找到的 MD 文件创建嵌入任务
                # --- [新增] 8.1 批量预处理 MD 文件，去除 "- " 开头的段落 ---
            if md_files_to_process_info:
                logger.info(f"Starting pre-processing for {len(md_files_to_process_info)} MD files to remove lines starting with '- '.")
                preprocessing_md_tasks = []
                async def _preprocess_single_md_file(pmid_key: str, file_path: str) -> Tuple[str, bool]:
                    """读取、处理并写回单个MD文件，返回PMID和处理成功状态"""
                    logger.debug(f"PMID {pmid_key}: Preprocessing MD file: {file_path}")
                    try:
                        # 使用 to_thread 执行同步文件 I/O 操作
                        def process_file_content_sync(path: str) -> bool:
                            try:
                                with open(path, 'r', encoding='utf-8') as f_read:
                                    lines = f_read.readlines()
                                
                                modified_lines = [line for line in lines if not line.startswith("- ")]
                                
                                # 只有当内容实际发生变化时才写回，以避免不必要的文件修改时间戳更新
                                # （虽然在这个场景下可能不那么重要，但通常是个好习惯）
                                # 注意：如果文件很大，这种比较可能消耗内存。
                                # 如果行数相同且内容相同，则可以认为没有变化。
                                # 一个更简单的方法是直接比较原始行数和修改后行数，
                                # 或者比较原始文本和修改后文本。
                                # 为简单起见，如果行数有变或检测到以 "- " 开头的行，就重写。
                                has_lines_to_remove = any(line.startswith("- ") for line in lines)
                                if has_lines_to_remove or len(lines) != len(modified_lines):
                                    with open(path, 'w', encoding='utf-8') as f_write:
                                        f_write.writelines(modified_lines)
                                    logger.info(f"PMID {pmid_key}: Successfully preprocessed (removed '- ' lines) and overwrote MD file: {path}")
                                else:
                                    logger.info(f"PMID {pmid_key}: No lines starting with '- ' found in {path}. File not modified.")
                                return True
                            except IOError as e:
                                logger.error(f"PMID {pmid_key}: IOError during MD file preprocessing for {path}: {e}", exc_info=True)
                                return False
                            except Exception as e_sync:
                                logger.error(f"PMID {pmid_key}: Unexpected error during synchronous MD file processing for {path}: {e_sync}", exc_info=True)
                                return False
                        
                        success = await asyncio.to_thread(process_file_content_sync, file_path)
                        return pmid_key, success
                    except Exception as e_async:
                        logger.error(f"PMID {pmid_key}: Error setting up preprocessing for MD file {file_path}: {e_async}", exc_info=True)
                        return pmid_key, False
                for pmid_iter, info_item_iter in md_files_to_process_info.items():
                    md_path_to_preprocess = info_item_iter.get("md_path")
                    if md_path_to_preprocess and os.path.exists(md_path_to_preprocess):
                         preprocessing_md_tasks.append(
                             asyncio.create_task(_preprocess_single_md_file(pmid_iter, md_path_to_preprocess))
                         )
                    else:
                        logger.warning(f"PMID {pmid_iter}: MD file path not found or file does not exist for preprocessing: {md_path_to_preprocess}")
                        # 可以选择将这个PMID从后续处理中移除，或者标记错误
                        # results["errors"].append(f"MD file for preprocessing not found for PMID {pmid_iter}")
                if preprocessing_md_tasks:
                    preprocessing_outcomes = await asyncio.gather(*preprocessing_md_tasks)
                    successfully_preprocessed_pmids = set()
                    for pmid_res, success_flag in preprocessing_outcomes:
                        if success_flag:
                            successfully_preprocessed_pmids.add(pmid_res)
                        else:
                            results["errors"].append(f"Failed to preprocess MD file for PMID {pmid_res}")
                    
                    # 可选：如果预处理失败，可以选择不将这些文件用于后续的嵌入。
                    # 当前代码会继续尝试嵌入所有在 md_files_to_process_info 中的文件，
                    # 即使预处理可能失败了。如果需要更严格的控制，
                    # 你可以创建一个新的字典，只包含成功预处理的PMID及其信息。
                    # 例如：
                    # temp_md_files_to_process_info = {
                    #    pmid: info for pmid, info in md_files_to_process_info.items()
                    #    if pmid in successfully_preprocessed_pmids
                    # }
                    # md_files_to_process_info = temp_md_files_to_process_info
                    # logger.info(f"{len(md_files_to_process_info)} MD files remain for embedding after preprocessing outcomes.")
                logger.info("Finished MD file preprocessing step.")
            else:
                logger.info("No MD files found in md_files_to_process_info, skipping MD preprocessing step.")        
            for pmid, process_info in md_files_to_process_info.items():
                 embedding_tasks.append(asyncio.create_task(
                     process_md_file_and_embed(
                         process_info["md_path"],
                         process_info["metadata"],
                         http_client,
                         determined_embedding_dimension
                     )
                 ))
            results["embedding_tasks_run"] = len(embedding_tasks)
            all_chunks_to_add = []
            embedded_papers_count = 0
            if embedding_tasks:
                logger.info(f"Starting {len(embedding_tasks)} MD file embedding tasks...")
                embedding_results = await asyncio.gather(*embedding_tasks, return_exceptions=True)
                logger.info("Finished MD embedding tasks.")
                for i, result_or_exc in enumerate(embedding_results):
                    # 尝试获取对应的 PMID (注意 gather 保留顺序)
                    pmid_for_log = list(md_files_to_process_info.keys())[i] if i < len(md_files_to_process_info) else f"embed_task_{i}"
                    if isinstance(result_or_exc, Exception):
                        error_msg = f"Error processing/embedding MD for PMID {pmid_for_log}: {type(result_or_exc).__name__} - {result_or_exc}"
                        logger.error(f"Exception in embedding task for PMID {pmid_for_log}:", exc_info=result_or_exc)
                        results["errors"].append(f"{error_msg} (Traceback logged)")
                    elif result_or_exc is None:
                        logger.warning(f"Processing/embedding task for PMID {pmid_for_log} returned None (likely file read error or no valid chunks/embeddings).")
                        results["errors"].append(f"Processing/embedding skipped or failed for PMID {pmid_for_log}")
                    elif isinstance(result_or_exc, list) and result_or_exc: # 确保是列表且非空
                         all_chunks_to_add.extend(result_or_exc)
                         embedded_papers_count += 1
                         logger.info(f"Successfully processed and embedded MD for PMID {pmid_for_log}, got {len(result_or_exc)} chunks.")
                    elif isinstance(result_or_exc, list) and not result_or_exc:
                         logger.warning(f"Embedding task for PMID {pmid_for_log} completed but yielded zero chunks.")
                         results["errors"].append(f"No embeddable chunks generated from MD for PMID {pmid_for_log}")
                    else:
                        logger.error(f"Unexpected result type '{type(result_or_exc)}' from embedding task for PMID {pmid_for_log}.")
                        results["errors"].append(f"Unexpected result type from embedding for PMID {pmid_for_log}: {type(result_or_exc).__name__}")
            else:
                logger.info("No MD files found to process for embedding.")
            results["papers_embedded"] = embedded_papers_count
            # --- 9. 批量添加文献内容块到 ChromaDB ---
            added_chunks_count = 0
            if all_chunks_to_add:
                # 过滤无效块 (双重检查)
                valid_chunks = [item for item in all_chunks_to_add if item.get('embedding') and len(item['embedding']) == determined_embedding_dimension]
                invalid_count = len(all_chunks_to_add) - len(valid_chunks)
                if invalid_count > 0:
                    logger.error(f"Found {invalid_count} chunks with invalid embeddings (None or wrong dimension). Skipping upsert for these.")
                    results["errors"].append(f"Skipped upserting {invalid_count} chunks due to invalid embeddings.")
                if valid_chunks:
                    final_chunk_ids = [item['id'] for item in valid_chunks]
                    final_chunk_embeddings = [item['embedding'] for item in valid_chunks]
                    final_chunk_metadatas = [item['metadata'] for item in valid_chunks]
                    final_chunk_documents = [item['document'] for item in valid_chunks]
                    logger.info(f"Preparing to add/update {len(final_chunk_ids)} valid document chunks in ChromaDB collection '{db_name}'...")
                    try:
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
                    except chromadb.errors.InvalidDimensionException as dim_err:
                        error_msg = f"FATAL: ChromaDB upsert failed due to dimension mismatch: {dim_err}."
                        logger.error(error_msg, exc_info=True); results["message"] = error_msg; results["errors"].append(f"{error_msg} (Traceback logged)"); results["success"] = False; results["chunks_embedded"] = 0
                    except Exception as e:
                        error_msg = f"Failed to upsert document chunks in ChromaDB: {e}"; logger.error(error_msg, exc_info=True); results["message"] = f"{error_msg} (Traceback logged)"; results["errors"].append(results["message"]); results["success"] = False; results["chunks_embedded"] = 0
                else:
                    logger.warning("No valid chunks with correct embeddings found to upsert after filtering.")
            else:
                logger.warning("No document chunks were generated or survived validation for embedding.")
            # --- 10. 设置最终状态和消息 ---
            if added_chunks_count > 0: # 只要成功嵌入了块，就认为主要目标达成
                 results["success"] = True
                 results["message"] = (f"PubMed found {results['papers_found']} papers with PMID. "
                                       f"Saved {results['files_saved_count']} content files. "
                                       f"Marker processed {results['marker_processed_success']}/{results['pdfs_moved_for_marker']} PDFs. "
                                       f"Successfully embedded {added_chunks_count} chunks from {results['papers_embedded']} papers "
                                       f"into collection '{db_name}'. "
                                       f"PMID-Abstracts entry saved: {results['pmid_abstract_entry_saved']}.")
            elif results["files_saved_count"] > 0 and not results["errors"]: # 文件保存了，但没有嵌入块，且无错误
                 results["success"] = True
                 results["message"] = (f"Processed {results['papers_found']} papers, saved {results['files_saved_count']} files. "
                                       f"Marker: {results['marker_processed_success']} success, {results['marker_processed_failure']} fail. "
                                       f"However, generated no valid chunks for embedding. "
                                       f"PMID-Abstracts entry saved: {results['pmid_abstract_entry_saved']}.")
            else: # 有错误，或完全没找到/保存文件
                 results["success"] = False
                 # 消息会在 finally 块中基于错误进一步完善
        except Exception as e: # 捕获 search_literature 函数主体中的任何未预料异常
            error_msg = f"Critical unhandled error in search_literature: {type(e).__name__} - {e}"
            logger.error(error_msg, exc_info=True)
            results["message"] = f"{error_msg}. Check server logs."
            results["errors"].append(results["message"])
            results["success"] = False
        finally:
            # 完善最终消息
            if results["errors"] and not results["success"]:
                 if results["message"] and ("error" in results["message"].lower() or "failed" in results["message"].lower()):
                      results["message"] += f" Encountered {len(results['errors'])} total errors. Check errors list and server logs."
                 else:
                      error_summary = "; ".join(list(set(results["errors"]))[:2]) + ('...' if len(results["errors"]) > 2 else '')
                      results["message"] = (f"Process finished with {len(results['errors'])} errors (e.g., {error_summary}). "
                                            f"Saved: {results['files_saved_count']}, Marker OK: {results['marker_processed_success']}, Embedded Chunks: {results['chunks_embedded']}. "
                                            f"Check errors list and logs.")
            elif not results["success"] and not results["message"]:
                  results["message"] = "Process finished with an unknown error. Check server logs."
            end_time = time.time()
            logger.info(f"Literature search finished for '{keyword}'. Time: {end_time - start_time:.2f}s. Success: {results['success']}. Message: {results['message']}")
        return results # 返回最终结果
# --- 其他工具函数 (get_combined_abstracts, search_text_from_chromadb) ---
# get_combined_abstracts 需要修改以解析 JSON
@mcp.tool()
async def get_combined_abstracts(db_name: str) -> str:
    """
    从指定的 ChromaDB 集合中获取预存的所有 PMID 及其摘要的合并信息。
    之前存储的是 JSON 字符串，这里会尝试解析并格式化输出。
    Args:
        db_name (str): 要查询的 ChromaDB 集合名称。
    Returns:
        str: 包含所有 PMID 和摘要的格式化文本，或错误/提示信息。
    """
    logger.info(f"Attempting to retrieve combined PMID-abstracts entry ('{ALL_ABSTRACTS_DOC_ID}') from ChromaDB collection: '{db_name}'")
    output_message = f"未能在数据库 '{db_name}' 中找到存储的 PMID-摘要 合集条目 ('{ALL_ABSTRACTS_DOC_ID}') 或其内容。"
    try:
        logger.debug(f"Connecting to ChromaDB at path: {CHROMA_DB_PATH}")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            collection = await asyncio.to_thread(chroma_client.get_collection, name=db_name)
            logger.info(f"Successfully accessed collection '{db_name}'.")
        except ValueError as e:
             if f"Collection {db_name} not found" in str(e) or "does not exist" in str(e).lower():
                 return f"错误：数据库集合 '{db_name}' 不存在。"
             else:
                  logger.error(f"ValueError accessing collection '{db_name}': {e}.", exc_info=True)
                  return f"无法访问数据库集合 '{db_name}' (值错误)。错误：{e}"
        except Exception as e:
            logger.error(f"Error accessing collection '{db_name}': {e}.", exc_info=True)
            return f"无法访问数据库集合 '{db_name}'。错误：{e}"
        logger.info(f"Fetching entry with ID '{ALL_ABSTRACTS_DOC_ID}' from collection '{db_name}', including metadatas.")
        try:
            collection_data = await asyncio.to_thread(
                collection.get,
                ids=[ALL_ABSTRACTS_DOC_ID],
                include=['metadatas']
            )
        except Exception as e:
             logger.warning(f"Could not get entry '{ALL_ABSTRACTS_DOC_ID}' from collection '{db_name}', likely does not exist: {e}")
             collection_data = {'ids': [], 'metadatas': []}
        retrieved_ids = collection_data.get('ids', [])
        retrieved_metadatas = collection_data.get('metadatas')
        if ALL_ABSTRACTS_DOC_ID in retrieved_ids and retrieved_metadatas and len(retrieved_metadatas) > 0:
            metadata_index = retrieved_ids.index(ALL_ABSTRACTS_DOC_ID)
            if metadata_index < len(retrieved_metadatas):
                 metadata_dict = retrieved_metadatas[metadata_index]
                 if metadata_dict:
                     # [修改] 提取 JSON 字符串并解析
                     json_data_str = metadata_dict.get(COMBINED_ABSTRACT_METADATA_KEY)
                     if json_data_str and isinstance(json_data_str, str):
                         try:
                             pmid_abstract_list = await asyncio.to_thread(json.loads, json_data_str)
                             if isinstance(pmid_abstract_list, list) and pmid_abstract_list:
                                 output_lines = ["--- Combined PMID and Abstracts ---"]
                                 for item in pmid_abstract_list:
                                     pmid = item.get('pmid', 'N/A')
                                     abstract = item.get('abstract', 'N/A')
                                     output_lines.append(f"\nPMID: {pmid}")
                                     output_lines.append(f"Abstract:\n{abstract}\n")
                                     output_lines.append("-" * 20)
                                 output_message = "\n".join(output_lines)
                                 logger.info(f"Successfully parsed and formatted {len(pmid_abstract_list)} PMID-abstract pairs from metadata.")
                             elif isinstance(pmid_abstract_list, list) and not pmid_abstract_list:
                                 output_message = "Combined PMID-abstracts entry found, but the list is empty."
                                 logger.warning(output_message)
                             else:
                                 output_message = f"Metadata for '{COMBINED_ABSTRACT_METADATA_KEY}' is not a valid JSON list."
                                 logger.warning(output_message)
                         except json.JSONDecodeError as json_err:
                             output_message = f"Metadata found, but failed to parse JSON content for key '{COMBINED_ABSTRACT_METADATA_KEY}': {json_err}"
                             logger.error(output_message)
                         except Exception as parse_err:
                             output_message = f"Unexpected error parsing JSON metadata: {parse_err}"
                             logger.error(output_message, exc_info=True)
                     else:
                         output_message = f"Metadata found for ID {ALL_ABSTRACTS_DOC_ID}, but key '{COMBINED_ABSTRACT_METADATA_KEY}' is missing or not a string."
                         logger.warning(output_message)
                 else:
                      logger.warning(f"Metadata dictionary retrieved but is empty/None for ID '{ALL_ABSTRACTS_DOC_ID}'.")
            else:
                 logger.error(f"Inconsistency found: ID {ALL_ABSTRACTS_DOC_ID} in ids list but not in metadatas list.")
        else:
            logger.warning(f"Entry with ID '{ALL_ABSTRACTS_DOC_ID}' not found in collection '{db_name}'.")
    except Exception as e:
        error_msg = f"An unexpected error occurred while retrieving combined PMID-abstracts from '{db_name}': {e}"
        logger.error(error_msg, exc_info=True)
        output_message = f"处理数据库 '{db_name}' 时发生意外错误：{e}"
    finally:
        logger.info(f"Finished retrieving combined PMID-abstracts from '{db_name}'.")
    return output_message



@mcp.tool()
async def get_reference(text_input: str, db_name: str) -> str:
    """
    识别输入文本中的PMID，按其首次出现的顺序编号，并从ChromaDB中检索文献信息，
    生成格式化的尾注列表。
    Args:
        text_input (str): 包含PMID的输入文本（例如综述）。
        db_name (str): 要查询的ChromaDB集合名称。
    Returns:
        str: 格式化的尾注列表，条目间有空行。如果出错或未找到信息，则返回提示信息。
    """
    logger.info(f"Initiating get_reference for db_name='{db_name}'. Processing input text of length {len(text_input)}.")
    # 在 search_literature 中定义的常量，确保在此处也可用或硬编码一致
    TITLE_PMID_DOI_DOC_ID = "title_pmid_doi"
    TITLE_PMID_DOI_METADATA_KEY = "title_pmid_doi_json"
    # 1. 按顺序从文本中识别PMID第一次出现的位置
    # 正则表达式查找独立的7到8位数字作为PMID
    # 使用 finditer 来获取匹配对象及其位置
    pmid_mentions = []
    for match in re.finditer(r'\b(\d{7,8})\b', text_input):
        pmid = match.group(1)
        start_index = match.start()
        pmid_mentions.append({'pmid': pmid, 'index': start_index})
    if not pmid_mentions:
        logger.warning("No PMIDs found in the input text.")
        return "在输入文本中没有识别到任何PMID。"
    # 根据出现位置排序
    pmid_mentions.sort(key=lambda x: x['index'])
    # 获取唯一的PMID，并保持其首次出现的顺序
    ordered_unique_pmids_in_text = []
    seen_pmids = set()
    for mention in pmid_mentions:
        if mention['pmid'] not in seen_pmids:
            ordered_unique_pmids_in_text.append(mention['pmid'])
            seen_pmids.add(mention['pmid'])
    if not ordered_unique_pmids_in_text:
        #理论上，如果pmid_mentions非空，这里也不会为空，但作为安全检查
        logger.warning("PMIDs were matched, but no unique PMIDs were extracted in order. This is unexpected.")
        return "未能从文本中提取有序的唯一PMID。"
    logger.info(f"Identified {len(ordered_unique_pmids_in_text)} unique PMIDs in order of appearance: {ordered_unique_pmids_in_text}")
    # 2. 从ChromaDB获取文献元数据（标题、PMID、DOI）
    master_literature_map = {} # 将存储 pmid -> {title, doi}
    try:
        logger.debug(f"Connecting to ChromaDB at path: {CHROMA_DB_PATH}")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            collection = await asyncio.to_thread(chroma_client.get_collection, name=db_name)
            logger.info(f"Successfully accessed collection '{db_name}'.")
        except ValueError as e:
            if f"Collection {db_name} not found" in str(e) or "does not exist" in str(e).lower():
                logger.error(f"ChromaDB collection '{db_name}' not found.")
                return f"错误：数据库集合 '{db_name}' 不存在。"
            logger.error(f"ValueError accessing collection '{db_name}': {e}.", exc_info=True)
            return f"无法访问数据库集合 '{db_name}' (值错误)。错误：{e}"
        except Exception as e:
            logger.error(f"Error accessing collection '{db_name}': {e}.", exc_info=True)
            return f"无法访问数据库集合 '{db_name}'。错误：{e}"
        logger.info(f"Fetching entry with ID '{TITLE_PMID_DOI_DOC_ID}' from collection '{db_name}' for master literature list.")
        collection_data = await asyncio.to_thread(
            collection.get,
            ids=[TITLE_PMID_DOI_DOC_ID],
            include=['metadatas']
        )
        retrieved_ids = collection_data.get('ids', [])
        retrieved_metadatas = collection_data.get('metadatas')
        if TITLE_PMID_DOI_DOC_ID in retrieved_ids and retrieved_metadatas and len(retrieved_metadatas) > 0:
            metadata_index = retrieved_ids.index(TITLE_PMID_DOI_DOC_ID)
            if metadata_index < len(retrieved_metadatas):
                metadata_dict = retrieved_metadatas[metadata_index]
                if metadata_dict:
                    json_data_str = metadata_dict.get(TITLE_PMID_DOI_METADATA_KEY)
                    if json_data_str and isinstance(json_data_str, str):
                        try:
                            literature_list = await asyncio.to_thread(json.loads, json_data_str)
                            if isinstance(literature_list, list):
                                for item in literature_list:
                                    if isinstance(item, dict) and 'pmid' in item:
                                        master_literature_map[item['pmid']] = {
                                            'title': item.get('title', 'N/A'),
                                            'doi': item.get('doi') # DOI可以是None
                                        }
                                logger.info(f"Successfully parsed {len(master_literature_map)} entries from master literature list.")
                            else:
                                logger.error(f"Master literature data (key: {TITLE_PMID_DOI_METADATA_KEY}) is not a valid JSON list.")
                                return f"错误：存储的文献主列表格式不正确（非列表）。"
                        except json.JSONDecodeError as json_err:
                            logger.error(f"Failed to parse JSON master literature list: {json_err}")
                            return f"错误：无法解析存储的文献主列表 (JSON解码失败)。"
                    else:
                        logger.warning(f"Master literature list key '{TITLE_PMID_DOI_METADATA_KEY}' not found or not a string in metadata for ID '{TITLE_PMID_DOI_DOC_ID}'.")
                else:
                    logger.warning(f"Metadata dictionary is empty for ID '{TITLE_PMID_DOI_DOC_ID}'.")
        
        if not master_literature_map:
            logger.warning(f"Master literature list (ID: {TITLE_PMID_DOI_DOC_ID}) could not be loaded from collection '{db_name}'. No references can be generated.")
            return f"错误：未能从数据库 '{db_name}' 加载文献主列表。无法生成参考文献。"
    except Exception as e:
        logger.error(f"An unexpected error occurred while retrieving master literature list from '{db_name}': {e}", exc_info=True)
        return f"处理数据库 '{db_name}' 时发生意外错误：{e}"
    # 3. 构建尾注列表
    endnotes_output = []
    current_reference_number = 1
    for pmid_in_text in ordered_unique_pmids_in_text:
        if pmid_in_text in master_literature_map:
            paper_details = master_literature_map[pmid_in_text]
            title = paper_details.get('title', 'N/A')
            doi = paper_details.get('doi') # Can be None
            doi_str = f"DOI: {doi}" if doi else "DOI: N/A" # 确保有DOI标签
            
            reference_entry = f"{current_reference_number}. {title}. ({doi_str}, PMID: {pmid_in_text})"
            endnotes_output.append(reference_entry)
            current_reference_number += 1
        else:
            # 如果文本中的PMID在主文献列表中找不到，记录警告，并可能添加一个占位符条目
            logger.warning(f"PMID {pmid_in_text} (from input text) not found in the master literature list from ChromaDB collection '{db_name}'.")
            reference_entry = f"{current_reference_number}. [文献信息缺失] 标题未在库中找到 (PMID: {pmid_in_text})"
            endnotes_output.append(reference_entry)
            current_reference_number += 1
            
    if not endnotes_output:
        logger.info("No reference entries were generated, possibly because PMIDs from text were not in the master list or no PMIDs were found.")
        return "未能生成尾注列表。可能原因：文本中的PMID未在文献库中找到，或文本中无PMID。"
    logger.info(f"Successfully generated {len(endnotes_output)} reference entries.")
    return "\n\n".join(endnotes_output)

    
@mcp.tool()
async def search_text_from_chromadb(
    db_name: str,
    reference_text: str,
    n_results: int = 5,
    delimiter: str = "\n"
) -> List[Dict[str, Any]]:
    """
    根据提供的参考文本（按指定分隔符拆分查询），在指定的 ChromaDB 集合中搜索相关文本块。
    检索完毕所有的输入文本片段后，将所有匹配的文本块整理为一个list，
    包括其来源pmid和完整的文本块内容，然后去除内容重复的文本块，作为输出。
    支持并发检索以提升速度。
    """
    start_time = time.time()
    logger.info(f"Starting text search in ChromaDB collection '{db_name}' for reference text. Output: List of unique blocks.")
    all_matched_blocks: List[Dict[str, Any]] = []
    if not reference_text or not reference_text.strip():
        logger.warning("Reference text is empty or contains only whitespace. Returning empty list.")
        return []
    try:
        logger.debug(f"Connecting to ChromaDB at path: {CHROMA_DB_PATH}")
        try:
            chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            collection = await asyncio.to_thread(chroma_client.get_collection, name=db_name)
            logger.info(f"Successfully accessed collection '{db_name}'.")
        except ValueError as e:
            if f"Collection {db_name} not found" in str(e) or "does not exist" in str(e).lower():
                error_msg = f"Collection '{db_name}' does not exist at path '{CHROMA_DB_PATH}'."
                logger.error(error_msg)
            else:
                error_msg = f"ValueError accessing collection '{db_name}': {e}."
                logger.error(error_msg, exc_info=True)
            return [] # 返回空列表表示错误或未找到集合
        except Exception as e:
            error_msg = f"Error accessing collection '{db_name}': {e}."
            logger.error(error_msg, exc_info=True)
            return [] # 返回空列表
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            segments = [segment.strip() for segment in reference_text.strip().split(delimiter) if segment.strip()]
            logger.info(f"Reference text split into {len(segments)} non-empty segments for querying using delimiter '{delimiter}'.")
            if not segments:
                logger.warning(f"Reference text after splitting by '{delimiter}' resulted in no valid query segments. Returning empty list.")
                return []

            # 新增：设置并发限制，例如同时处理3个片段
            CONCURRENT_SEARCH_LIMIT = 3
            semaphore = asyncio.Semaphore(CONCURRENT_SEARCH_LIMIT)
            search_tasks = []

            async def process_segment(i: int, query_segment: str):
                async with semaphore:
                    logger.info(f"Processing query segment {i+1}/{len(segments)}: \"{query_segment[:100]}...\"")
                    query_embedding: Optional[List[float]] = None
                    try:
                        query_embedding = await _get_single_embedding(
                            query_segment, SILICONFLOW_API_KEY, SILICONFLOW_API_URL, SILICONFLOW_EMBEDDING_MODEL, http_client
                        )
                        if query_embedding is None:
                            logger.error(f"Failed to get embedding for query segment: \"{query_segment[:100]}...\" Skipping this segment.")
                            return []
                    except Exception as embed_err:
                        logger.error(f"Error getting embedding for query segment \"{query_segment[:100]}...\": {embed_err}", exc_info=True)
                        return [] # 跳过这个查询片段

                    try:
                        logger.debug(f"Querying ChromaDB collection '{db_name}' with embedding for segment {i+1}, n_results={n_results}.")
                        query_results = await asyncio.to_thread(
                            collection.query, query_embeddings=[query_embedding],
                            n_results=min(n_results, 50), # 限制最大结果数
                            include=['metadatas', 'documents']
                        )
                        logger.debug(f"ChromaDB query completed for segment {i+1}.")
                    except Exception as query_err:
                        logger.error(f"Error querying ChromaDB for segment \"{query_segment[:100]}...\": {query_err}", exc_info=True)
                        return [] # 跳过这个查询片段

                    matched_blocks = []
                    metadatas_list = query_results.get('metadatas', [[]])[0]
                    documents_list = query_results.get('documents', [[]])[0]
                    if not documents_list:
                        logger.info(f"No relevant documents found in '{db_name}' for query segment: \"{query_segment[:100]}...\"")
                        return []
                    logger.info(f"Found {len(documents_list)} potentially relevant content blocks for segment {i+1}.")
                    for metadata, document_content in zip(metadatas_list, documents_list):
                        pmid = metadata.get('pmid')
                        if pmid and document_content:
                            matched_blocks.append({
                                "pmid": str(pmid),
                                "text_content": str(document_content)
                            })
                        else:
                            logger.warning(f"Skipping a block due to missing pmid or content. Metadata: {metadata}")
                    return matched_blocks

            # 创建任务列表，处理所有片段
            for i, segment in enumerate(segments):
                search_tasks.append(process_segment(i, segment))

            # 并发执行所有任务
            results = await asyncio.gather(*search_tasks)
            for segment_results in results:
                all_matched_blocks.extend(segment_results)

            # 去重处理
            unique_blocks: List[Dict[str, Any]] = []
            seen_contents = set()
            if all_matched_blocks:
                logger.info(f"Collected {len(all_matched_blocks)} raw blocks from all query segments. Starting deduplication.")
                for block in all_matched_blocks:
                    content_to_check = block.get("text_content")
                    if content_to_check and content_to_check not in seen_contents:
                        unique_blocks.append(block)
                        seen_contents.add(content_to_check)
                logger.info(f"Deduplication complete. {len(unique_blocks)} unique blocks remaining.")
            else:
                logger.info("No blocks were matched across all query segments.")
            return unique_blocks
    except Exception as e:
        error_msg = f"An unexpected critical error occurred during search_text_from_chromadb for collection '{db_name}': {e}"
        logger.error(error_msg, exc_info=True)
        return [] # 发生未捕获的严重错误时返回空列表
    finally:
        end_time = time.time()
        logger.info(f"Finished text search in ChromaDB '{db_name}'. Time taken: {end_time - start_time:.2f} seconds.")


# --- 主程序入口 ---
if __name__ == "__main__":
    logger.info("Starting Literature Search MCP Server (via stdio)...")
    # ... (ChromaDB 路径检查保持不变) ...
    try:
        if not os.path.exists(CHROMA_DB_PATH):
             logger.info(f"ChromaDB path '{CHROMA_DB_PATH}' does not exist. Attempting to create it.")
             os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        else:
             logger.info(f"ChromaDB path '{CHROMA_DB_PATH}' already exists.")
        test_file_path = os.path.join(CHROMA_DB_PATH, ".write_test")
        with open(test_file_path, "w") as f: f.write("test")
        os.remove(test_file_path)
        logger.info(f"Successfully tested write access to ChromaDB path: {CHROMA_DB_PATH}")
    except OSError as e:
        logger.error(f"FATAL: Error accessing or creating ChromaDB path '{CHROMA_DB_PATH}': {e}. Check permissions.", exc_info=True)
        sys.exit(1)
    except Exception as e:
         logger.error(f"FATAL: Unexpected error during ChromaDB path check '{CHROMA_DB_PATH}': {e}.", exc_info=True)
         sys.exit(1)

    mcp.run()
    logger.info("MCP Server stopped.")

