from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen, urlretrieve
import xml.etree.ElementTree as ET

import wikipediaapi


def extract_medical_entities(text: str, limit: int = 10) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "of",
        "on",
        "or",
        "patient",
        "the",
        "with",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)
    unique: list[str] = []
    for token in tokens:
        normalized = token.lower()
        if normalized in stopwords or normalized in unique:
            continue
        unique.append(normalized)
        if len(unique) >= limit:
            break
    return unique


def download_url(url: str, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, target)
    return target


def fetch_wikipedia_pages(entities: Iterable[str], top_n: int = 5) -> dict[str, str]:
    wiki = wikipediaapi.Wikipedia("medqa-rag/0.1", "en")
    pages: dict[str, str] = {}
    for entity in list(entities)[:top_n]:
        page = wiki.page(entity)
        if page.exists():
            pages[entity] = page.text
    return pages


def fetch_pubmed_abstracts(
    queries: Iterable[str],
    max_results: int = 20,
    email: str | None = None,
    api_key: str | None = None,
    sleep_seconds: float = 0.34,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for query in queries:
        search_params = {
            "db": "pubmed",
            "retmax": str(max_results),
            "retmode": "json",
            "term": query,
        }
        if email:
            search_params["email"] = email
        if api_key:
            search_params["api_key"] = api_key
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urlencode(search_params)
        with urlopen(search_url) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ids = payload.get("esearchresult", {}).get("idlist", [])
        if not ids:
            continue

        fetch_params = {
            "db": "pubmed",
            "retmode": "xml",
            "id": ",".join(ids),
        }
        if email:
            fetch_params["email"] = email
        if api_key:
            fetch_params["api_key"] = api_key
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urlencode(fetch_params)
        with urlopen(fetch_url) as response:
            xml_text = response.read().decode("utf-8")

        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID", default="")
            title = article.findtext(".//ArticleTitle", default="")
            abstract = " ".join(
                text.strip()
                for text in article.findall(".//Abstract/AbstractText")
                if text.text
            ).strip()
            if abstract:
                results.append({"query": query, "pmid": pmid, "title": title, "abstract": abstract})
        time.sleep(sleep_seconds)
    return results


def save_jsonl(records: Iterable[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
