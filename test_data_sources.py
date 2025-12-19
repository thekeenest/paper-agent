#!/usr/bin/env python3
"""
Тестирование интеграций с внешними источниками данных.

Запуск:
    python test_data_sources.py

Тестирует:
    1. ArXiv API
    2. Semantic Scholar API
    3. OpenAlex API
    4. ROR (Research Organization Registry)
    5. DataSourceRouter
"""

import os
import sys
from pathlib import Path

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

from src.data_sources import (
    ArxivClient,
    SemanticScholarClient,
    OpenAlexClient,
    RORLookup,
    DataSourceRouter,
    DataSourceType,
    SearchParams
)


def test_arxiv():
    """Тест ArXiv API"""
    print("\n" + "="*60)
    print("TESTING: ArXiv API")
    print("="*60)
    
    client = ArxivClient()
    
    # Тест 1: Поиск
    print("\n[Test 1] Search for 'machine learning' in cs.AI...")
    params = SearchParams(
        query="cat:cs.AI",
        max_results=3
    )
    papers = client.search(params)
    
    print(f"  Found: {len(papers)} papers")
    if papers:
        paper = papers[0]
        print(f"  First paper: {paper.title[:60]}...")
        print(f"  ArXiv ID: {paper.arxiv_id}")
        print(f"  Categories: {paper.categories}")
        print(f"  PDF URL: {paper.pdf_url}")
        print(f"  Authors: {len(paper.authors)}")
        print(f"  Has affiliations: {any(a.raw_affiliation for a in paper.authors)}")
    
    # Тест 2: Получение по ID
    print("\n[Test 2] Get paper by ID...")
    if papers:
        paper = client.get_paper(papers[0].arxiv_id)
        if paper:
            print(f"  Retrieved: {paper.title[:60]}...")
        else:
            print("  ERROR: Could not retrieve paper")
    
    print(f"\nTotal requests: {client.get_request_count()}")
    print(f"Supports affiliations: {client.supports_affiliations()}")
    print(f"Supports citations: {client.supports_citations()}")
    
    return len(papers) > 0


def test_semantic_scholar():
    """Тест Semantic Scholar API"""
    print("\n" + "="*60)
    print("TESTING: Semantic Scholar API")
    print("="*60)
    
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    print(f"API Key configured: {'Yes' if api_key else 'No'}")
    
    client = SemanticScholarClient()
    
    # Тест 1: Поиск
    print("\n[Test 1] Search for 'transformer attention'...")
    params = SearchParams(
        query="transformer attention neural network",
        max_results=3
    )
    
    try:
        papers = client.search(params)
        print(f"  Found: {len(papers)} papers")
        
        if papers:
            paper = papers[0]
            print(f"  First paper: {paper.title[:60]}...")
            print(f"  ID: {paper.arxiv_id}")
            print(f"  Authors: {len(paper.authors)}")
            
            # Проверяем аффилиации
            affiliations_found = sum(1 for a in paper.authors if a.raw_affiliation)
            print(f"  Authors with affiliations: {affiliations_found}/{len(paper.authors)}")
            
            if paper.authors:
                for author in paper.authors[:3]:
                    print(f"    - {author.name}: {author.raw_affiliation or 'N/A'}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    
    # Тест 2: Получение по ArXiv ID
    print("\n[Test 2] Get paper by ArXiv ID (2401.12345)...")
    try:
        paper = client.get_paper("2401.12345")
        if paper:
            print(f"  Retrieved: {paper.title[:60]}...")
            print(f"  Authors: {len(paper.authors)}")
        else:
            print("  Paper not found (expected for random ID)")
    except Exception as e:
        print(f"  Note: {e}")
    
    # Тест 3: Получение автора
    print("\n[Test 3] Get author info...")
    try:
        # Yann LeCun's Semantic Scholar ID
        author = client.get_author("1688681")
        if author:
            print(f"  Name: {author['name']}")
            print(f"  Affiliations: {author['affiliations']}")
            print(f"  h-index: {author['h_index']}")
    except Exception as e:
        print(f"  Note: {e}")
    
    print(f"\nTotal requests: {client.get_request_count()}")
    print(f"Supports affiliations: {client.supports_affiliations()}")
    print(f"Supports citations: {client.supports_citations()}")
    
    client.close()
    return True


def test_openalex():
    """Тест OpenAlex API"""
    print("\n" + "="*60)
    print("TESTING: OpenAlex API")
    print("="*60)
    
    email = os.getenv("OPENALEX_EMAIL")
    print(f"Email configured (polite pool): {'Yes' if email else 'No'}")
    
    client = OpenAlexClient()
    
    # Тест 1: Поиск
    print("\n[Test 1] Search for 'deep learning'...")
    params = SearchParams(
        query="deep learning neural network",
        max_results=3
    )
    
    try:
        papers = client.search(params)
        print(f"  Found: {len(papers)} papers")
        
        if papers:
            paper = papers[0]
            print(f"  First paper: {paper.title[:60]}...")
            print(f"  ID: {paper.arxiv_id}")
            print(f"  Year: {paper.published_date}")
            print(f"  Authors: {len(paper.authors)}")
            
            # Проверяем аффилиации
            affiliations_found = sum(1 for a in paper.authors if a.raw_affiliation)
            print(f"  Authors with affiliations: {affiliations_found}/{len(paper.authors)}")
            
            if paper.authors:
                for author in paper.authors[:3]:
                    aff = author.normalized_affiliation or author.raw_affiliation or 'N/A'
                    print(f"    - {author.name}: {aff[:40]}... [{author.country_code}]")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    
    # Тест 2: Получение организации
    print("\n[Test 2] Get institution info (MIT)...")
    try:
        inst = client.get_institution("I63966007")  # MIT's OpenAlex ID
        if inst:
            print(f"  Name: {inst['name']}")
            print(f"  Country: {inst['country']}")
            print(f"  Type: {inst['type']}")
            print(f"  Works count: {inst['works_count']}")
    except Exception as e:
        print(f"  Note: {e}")
    
    print(f"\nTotal requests: {client.get_request_count()}")
    print(f"Supports affiliations: {client.supports_affiliations()}")
    print(f"Supports citations: {client.supports_citations()}")
    
    client.close()
    return True


def test_ror():
    """Тест ROR API"""
    print("\n" + "="*60)
    print("TESTING: ROR (Research Organization Registry)")
    print("="*60)
    
    ror = RORLookup()
    
    test_orgs = [
        "Massachusetts Institute of Technology",
        "MIT",
        "Google Research",
        "Stanford University",
        "Tsinghua University",
        "DeepMind",
        "INRIA"
    ]
    
    print("\n[Test] Looking up organizations...")
    for org_name in test_orgs:
        result = ror.lookup(org_name)
        if result:
            print(f"\n  '{org_name}':")
            print(f"    Canonical: {result['name']}")
            print(f"    Country: {result['country']} ({result['country_code']})")
            print(f"    Type: {result['type']}")
            print(f"    Confidence: {result['confidence']:.2f}")
            print(f"    ROR ID: {result['ror_id']}")
        else:
            print(f"\n  '{org_name}': NOT FOUND")
    
    print(f"\nTotal requests: {ror.get_request_count()}")
    
    ror.close()
    return True


def test_router():
    """Тест DataSourceRouter"""
    print("\n" + "="*60)
    print("TESTING: DataSourceRouter")
    print("="*60)
    
    router = DataSourceRouter(
        default_source=DataSourceType.ARXIV,
        enable_ror=True
    )
    
    # Тест 1: Поиск в ArXiv
    print("\n[Test 1] Search in ArXiv...")
    papers = router.search(
        query="cat:cs.AI",
        source="arxiv",
        max_results=3
    )
    print(f"  Found: {len(papers)} papers from ArXiv")
    
    # Тест 2: Поиск в Semantic Scholar
    print("\n[Test 2] Search in Semantic Scholar...")
    papers = router.search(
        query="large language models",
        source="semantic_scholar",
        max_results=3
    )
    print(f"  Found: {len(papers)} papers from Semantic Scholar")
    if papers:
        affiliations_count = sum(1 for p in papers for a in p.authors if a.raw_affiliation)
        print(f"  Total affiliations found: {affiliations_count}")
    
    # Тест 3: Поиск в OpenAlex
    print("\n[Test 3] Search in OpenAlex...")
    papers = router.search(
        query="machine learning",
        source="openalex",
        max_results=3
    )
    print(f"  Found: {len(papers)} papers from OpenAlex")
    if papers:
        affiliations_count = sum(1 for p in papers for a in p.authors if a.raw_affiliation)
        print(f"  Total affiliations found: {affiliations_count}")
    
    # Тест 4: Автоопределение источника
    print("\n[Test 4] Auto-detect source...")
    papers = router.search(
        query="cat:cs.LG",
        max_results=2
    )
    print(f"  Auto-detected ArXiv for 'cat:cs.LG' query")
    
    # Тест 5: Обогащение данных
    print("\n[Test 5] Enrichment with ROR normalization...")
    if papers:
        paper = papers[0]
        enriched = router.enrich_paper(paper)
        normalized_count = sum(1 for a in enriched.authors if a.normalized_affiliation)
        print(f"  Normalized affiliations: {normalized_count}/{len(enriched.authors)}")
    
    # Статистика
    stats = router.get_stats()
    print(f"\nRouter stats: {stats}")
    
    router.close()
    return True


def main():
    print("\n" + "#"*60)
    print("#  DATA SOURCES INTEGRATION TEST")
    print("#"*60)
    
    results = {}
    
    # Тестируем каждый источник
    results["ArXiv"] = test_arxiv()
    results["Semantic Scholar"] = test_semantic_scholar()
    results["OpenAlex"] = test_openalex()
    results["ROR"] = test_ror()
    results["DataSourceRouter"] = test_router()
    
    # Итоги
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    all_passed = True
    for source, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {source}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed! ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
