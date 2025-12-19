"""
База знаний организаций для нормализации.

Содержит mapping вариантов написания к каноническим названиям.
"""

from typing import Dict, Any

# Структура записи:
# {
#     "canonical": str,      # Каноническое название
#     "variants": List[str], # Варианты написания
#     "country": str,        # Страна
#     "country_code": str,   # Код страны ISO
#     "type": str,           # Тип: university, company, research_institute
#     "parent": str,         # Родительская организация (опционально)
#     "aliases": List[str],  # Краткие названия/аббревиатуры
# }

ORGANIZATION_KB: Dict[str, Dict[str, Any]] = {
    # ==================== BIG TECH ====================
    "google": {
        "canonical": "Google",
        "variants": [
            "Google Research",
            "Google Brain", 
            "Google AI",
            "Google Inc.",
            "Google LLC",
            "Google DeepMind",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "company",
        "parent": "Alphabet Inc.",
        "aliases": ["GOOG"]
    },
    "deepmind": {
        "canonical": "DeepMind",
        "variants": [
            "DeepMind Technologies",
            "DeepMind Technologies Limited",
            "Google DeepMind",
        ],
        "country": "United Kingdom",
        "country_code": "GB",
        "type": "company",
        "parent": "Alphabet Inc.",
        "aliases": []
    },
    "meta": {
        "canonical": "Meta",
        "variants": [
            "Meta AI",
            "Meta Platforms",
            "Meta Platforms, Inc.",
            "Facebook",
            "Facebook AI Research",
            "Facebook AI",
            "FAIR",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "company",
        "aliases": ["META"]
    },
    "microsoft": {
        "canonical": "Microsoft",
        "variants": [
            "Microsoft Research",
            "Microsoft Research Asia",
            "Microsoft Research Lab",
            "MSR",
            "Microsoft Corporation",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "company",
        "aliases": ["MSFT"]
    },
    "openai": {
        "canonical": "OpenAI",
        "variants": [
            "OpenAI LP",
            "OpenAI Inc.",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "company",
        "aliases": []
    },
    "anthropic": {
        "canonical": "Anthropic",
        "variants": [
            "Anthropic PBC",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "company",
        "aliases": []
    },
    "nvidia": {
        "canonical": "NVIDIA",
        "variants": [
            "NVIDIA Corporation",
            "NVIDIA Research",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "company",
        "aliases": ["NVDA"]
    },
    "amazon": {
        "canonical": "Amazon",
        "variants": [
            "Amazon Web Services",
            "AWS",
            "Amazon Science",
            "Amazon Research",
            "Amazon.com",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "company",
        "aliases": ["AMZN"]
    },
    "apple": {
        "canonical": "Apple",
        "variants": [
            "Apple Inc.",
            "Apple Machine Learning Research",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "company",
        "aliases": ["AAPL"]
    },
    "ibm": {
        "canonical": "IBM",
        "variants": [
            "IBM Research",
            "IBM Corporation",
            "International Business Machines",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "company",
        "aliases": []
    },
    "tencent": {
        "canonical": "Tencent",
        "variants": [
            "Tencent AI Lab",
            "Tencent Holdings",
        ],
        "country": "China",
        "country_code": "CN",
        "type": "company",
        "aliases": []
    },
    "alibaba": {
        "canonical": "Alibaba",
        "variants": [
            "Alibaba Group",
            "Alibaba DAMO Academy",
            "DAMO Academy",
        ],
        "country": "China",
        "country_code": "CN",
        "type": "company",
        "aliases": ["BABA"]
    },
    "baidu": {
        "canonical": "Baidu",
        "variants": [
            "Baidu Research",
            "Baidu Inc.",
        ],
        "country": "China",
        "country_code": "CN",
        "type": "company",
        "aliases": []
    },
    "bytedance": {
        "canonical": "ByteDance",
        "variants": [
            "ByteDance AI Lab",
            "TikTok",
        ],
        "country": "China",
        "country_code": "CN",
        "type": "company",
        "aliases": []
    },
    "huawei": {
        "canonical": "Huawei",
        "variants": [
            "Huawei Technologies",
            "Huawei Noah's Ark Lab",
        ],
        "country": "China",
        "country_code": "CN",
        "type": "company",
        "aliases": []
    },
    "samsung": {
        "canonical": "Samsung",
        "variants": [
            "Samsung Research",
            "Samsung Electronics",
            "Samsung AI Center",
        ],
        "country": "South Korea",
        "country_code": "KR",
        "type": "company",
        "aliases": []
    },
    
    # ==================== US UNIVERSITIES ====================
    "stanford": {
        "canonical": "Stanford University",
        "variants": [
            "Stanford",
            "Stanford CS",
            "Stanford NLP",
            "Stanford AI Lab",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "university",
        "aliases": []
    },
    "mit": {
        "canonical": "Massachusetts Institute of Technology",
        "variants": [
            "MIT",
            "M.I.T.",
            "MIT CSAIL",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "university",
        "aliases": ["MIT"]
    },
    "berkeley": {
        "canonical": "University of California, Berkeley",
        "variants": [
            "UC Berkeley",
            "UCB",
            "Berkeley",
            "Berkeley AI Research",
            "BAIR",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "university",
        "aliases": []
    },
    "cmu": {
        "canonical": "Carnegie Mellon University",
        "variants": [
            "CMU",
            "Carnegie Mellon",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "university",
        "aliases": ["CMU"]
    },
    "harvard": {
        "canonical": "Harvard University",
        "variants": [
            "Harvard",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "university",
        "aliases": []
    },
    "princeton": {
        "canonical": "Princeton University",
        "variants": [
            "Princeton",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "university",
        "aliases": []
    },
    "caltech": {
        "canonical": "California Institute of Technology",
        "variants": [
            "Caltech",
            "Cal Tech",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "university",
        "aliases": ["Caltech"]
    },
    "cornell": {
        "canonical": "Cornell University",
        "variants": [
            "Cornell",
            "Cornell Tech",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "university",
        "aliases": []
    },
    "nyu": {
        "canonical": "New York University",
        "variants": [
            "NYU",
            "N.Y.U.",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "university",
        "aliases": ["NYU"]
    },
    "ucla": {
        "canonical": "University of California, Los Angeles",
        "variants": [
            "UCLA",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "university",
        "aliases": ["UCLA"]
    },
    "uw": {
        "canonical": "University of Washington",
        "variants": [
            "UW",
            "U Washington",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "university",
        "aliases": []
    },
    "uiuc": {
        "canonical": "University of Illinois Urbana-Champaign",
        "variants": [
            "UIUC",
            "University of Illinois",
            "UIUC CS",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "university",
        "aliases": ["UIUC"]
    },
    "gatech": {
        "canonical": "Georgia Institute of Technology",
        "variants": [
            "Georgia Tech",
            "GaTech",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "university",
        "aliases": []
    },
    
    # ==================== UK UNIVERSITIES ====================
    "oxford": {
        "canonical": "University of Oxford",
        "variants": [
            "Oxford",
            "Oxford University",
        ],
        "country": "United Kingdom",
        "country_code": "GB",
        "type": "university",
        "aliases": []
    },
    "cambridge": {
        "canonical": "University of Cambridge",
        "variants": [
            "Cambridge",
            "Cambridge University",
        ],
        "country": "United Kingdom",
        "country_code": "GB",
        "type": "university",
        "aliases": []
    },
    "imperial": {
        "canonical": "Imperial College London",
        "variants": [
            "Imperial",
            "Imperial College",
        ],
        "country": "United Kingdom",
        "country_code": "GB",
        "type": "university",
        "aliases": []
    },
    "ucl": {
        "canonical": "University College London",
        "variants": [
            "UCL",
        ],
        "country": "United Kingdom",
        "country_code": "GB",
        "type": "university",
        "aliases": ["UCL"]
    },
    "edinburgh": {
        "canonical": "University of Edinburgh",
        "variants": [
            "Edinburgh",
        ],
        "country": "United Kingdom",
        "country_code": "GB",
        "type": "university",
        "aliases": []
    },
    
    # ==================== CHINESE UNIVERSITIES ====================
    "tsinghua": {
        "canonical": "Tsinghua University",
        "variants": [
            "Tsinghua",
            "THU",
        ],
        "country": "China",
        "country_code": "CN",
        "type": "university",
        "aliases": []
    },
    "peking": {
        "canonical": "Peking University",
        "variants": [
            "Peking",
            "PKU",
            "Beijing University",
        ],
        "country": "China",
        "country_code": "CN",
        "type": "university",
        "aliases": ["PKU"]
    },
    "zju": {
        "canonical": "Zhejiang University",
        "variants": [
            "Zhejiang",
            "ZJU",
        ],
        "country": "China",
        "country_code": "CN",
        "type": "university",
        "aliases": ["ZJU"]
    },
    "sjtu": {
        "canonical": "Shanghai Jiao Tong University",
        "variants": [
            "SJTU",
            "Shanghai Jiao Tong",
        ],
        "country": "China",
        "country_code": "CN",
        "type": "university",
        "aliases": ["SJTU"]
    },
    "fudan": {
        "canonical": "Fudan University",
        "variants": [
            "Fudan",
        ],
        "country": "China",
        "country_code": "CN",
        "type": "university",
        "aliases": []
    },
    "cuhk": {
        "canonical": "The Chinese University of Hong Kong",
        "variants": [
            "CUHK",
            "Chinese University of Hong Kong",
        ],
        "country": "Hong Kong",
        "country_code": "HK",
        "type": "university",
        "aliases": ["CUHK"]
    },
    "hku": {
        "canonical": "The University of Hong Kong",
        "variants": [
            "HKU",
            "University of Hong Kong",
        ],
        "country": "Hong Kong",
        "country_code": "HK",
        "type": "university",
        "aliases": ["HKU"]
    },
    
    # ==================== CANADIAN UNIVERSITIES ====================
    "toronto": {
        "canonical": "University of Toronto",
        "variants": [
            "Toronto",
            "U of T",
            "UofT",
        ],
        "country": "Canada",
        "country_code": "CA",
        "type": "university",
        "aliases": []
    },
    "mila": {
        "canonical": "Mila - Quebec AI Institute",
        "variants": [
            "Mila",
            "MILA",
        ],
        "country": "Canada",
        "country_code": "CA",
        "type": "research_institute",
        "aliases": []
    },
    "mcgill": {
        "canonical": "McGill University",
        "variants": [
            "McGill",
        ],
        "country": "Canada",
        "country_code": "CA",
        "type": "university",
        "aliases": []
    },
    "waterloo": {
        "canonical": "University of Waterloo",
        "variants": [
            "Waterloo",
            "UWaterloo",
        ],
        "country": "Canada",
        "country_code": "CA",
        "type": "university",
        "aliases": []
    },
    
    # ==================== EU UNIVERSITIES ====================
    "ethz": {
        "canonical": "ETH Zurich",
        "variants": [
            "ETH",
            "ETHZ",
            "ETH Zürich",
        ],
        "country": "Switzerland",
        "country_code": "CH",
        "type": "university",
        "aliases": ["ETH"]
    },
    "epfl": {
        "canonical": "EPFL",
        "variants": [
            "École Polytechnique Fédérale de Lausanne",
        ],
        "country": "Switzerland",
        "country_code": "CH",
        "type": "university",
        "aliases": ["EPFL"]
    },
    "mpii": {
        "canonical": "Max Planck Institute for Informatics",
        "variants": [
            "MPI",
            "Max Planck",
            "MPI Informatics",
        ],
        "country": "Germany",
        "country_code": "DE",
        "type": "research_institute",
        "aliases": []
    },
    "tum": {
        "canonical": "Technical University of Munich",
        "variants": [
            "TUM",
            "TU Munich",
        ],
        "country": "Germany",
        "country_code": "DE",
        "type": "university",
        "aliases": ["TUM"]
    },
    "inria": {
        "canonical": "Inria",
        "variants": [
            "INRIA",
        ],
        "country": "France",
        "country_code": "FR",
        "type": "research_institute",
        "aliases": []
    },
    
    # ==================== ASIAN UNIVERSITIES ====================
    "kaist": {
        "canonical": "KAIST",
        "variants": [
            "Korea Advanced Institute of Science and Technology",
        ],
        "country": "South Korea",
        "country_code": "KR",
        "type": "university",
        "aliases": ["KAIST"]
    },
    "snu": {
        "canonical": "Seoul National University",
        "variants": [
            "SNU",
        ],
        "country": "South Korea",
        "country_code": "KR",
        "type": "university",
        "aliases": ["SNU"]
    },
    "tokyo": {
        "canonical": "The University of Tokyo",
        "variants": [
            "University of Tokyo",
            "UTokyo",
        ],
        "country": "Japan",
        "country_code": "JP",
        "type": "university",
        "aliases": []
    },
    "nus": {
        "canonical": "National University of Singapore",
        "variants": [
            "NUS",
        ],
        "country": "Singapore",
        "country_code": "SG",
        "type": "university",
        "aliases": ["NUS"]
    },
    "ntu_sg": {
        "canonical": "Nanyang Technological University",
        "variants": [
            "NTU Singapore",
            "NTU",
        ],
        "country": "Singapore",
        "country_code": "SG",
        "type": "university",
        "aliases": []
    },
    
    # ==================== RUSSIAN UNIVERSITIES ====================
    "mipt": {
        "canonical": "Moscow Institute of Physics and Technology",
        "variants": [
            "MIPT",
            "МФТИ",
            "Phystech",
            "Физтех",
        ],
        "country": "Russia",
        "country_code": "RU",
        "type": "university",
        "aliases": ["MIPT", "МФТИ"]
    },
    "hse": {
        "canonical": "HSE University",
        "variants": [
            "Higher School of Economics",
            "HSE",
            "ВШЭ",
        ],
        "country": "Russia",
        "country_code": "RU",
        "type": "university",
        "aliases": ["HSE"]
    },
    "msu": {
        "canonical": "Lomonosov Moscow State University",
        "variants": [
            "MSU",
            "Moscow State University",
            "МГУ",
        ],
        "country": "Russia",
        "country_code": "RU",
        "type": "university",
        "aliases": ["MSU", "МГУ"]
    },
    "skoltech": {
        "canonical": "Skolkovo Institute of Science and Technology",
        "variants": [
            "Skoltech",
            "Сколтех",
        ],
        "country": "Russia",
        "country_code": "RU",
        "type": "university",
        "aliases": ["Skoltech"]
    },
    
    # ==================== RESEARCH LABS ====================
    "allen_ai": {
        "canonical": "Allen Institute for AI",
        "variants": [
            "AI2",
            "Allen AI",
        ],
        "country": "United States",
        "country_code": "US",
        "type": "research_institute",
        "aliases": ["AI2"]
    },
}


def get_all_variants() -> Dict[str, str]:
    """
    Получить mapping всех вариантов написания к ключам KB.
    
    Returns:
        Dict[variant_lowercase, kb_key]
    """
    result = {}
    for key, data in ORGANIZATION_KB.items():
        # Добавляем ключ
        result[key] = key
        # Добавляем canonical
        result[data["canonical"].lower()] = key
        # Добавляем все варианты
        for variant in data["variants"]:
            result[variant.lower()] = key
        # Добавляем алиасы
        for alias in data.get("aliases", []):
            result[alias.lower()] = key
    
    return result


# Pre-computed lookup table
VARIANT_LOOKUP = get_all_variants()


def lookup_organization(text: str) -> Dict[str, Any] | None:
    """
    Найти организацию по тексту.
    
    Args:
        text: Текст для поиска (название организации)
    
    Returns:
        Данные организации из KB или None
    """
    text_lower = text.lower().strip()
    
    # Точное совпадение
    if text_lower in VARIANT_LOOKUP:
        key = VARIANT_LOOKUP[text_lower]
        return ORGANIZATION_KB[key]
    
    # Поиск подстроки
    for variant, key in VARIANT_LOOKUP.items():
        if variant in text_lower or text_lower in variant:
            return ORGANIZATION_KB[key]
    
    return None
