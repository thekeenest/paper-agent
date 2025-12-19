"""
ROR (Research Organization Registry) Lookup

Интеграция с ROR API для нормализации названий организаций.
https://ror.readme.io/docs/rest-api
"""

import time
from typing import Optional, Dict, Any, List
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from rapidfuzz import fuzz

from ..models import OrganizationType


class RORLookup:
    """
    Клиент для поиска организаций в Research Organization Registry.
    
    ROR - это глобальный реестр исследовательских организаций с уникальными
    идентификаторами. Используется OpenAlex и другими платформами.
    
    API Documentation: https://ror.readme.io/docs/rest-api
    
    Особенности:
    - Полностью бесплатный
    - Нет жёстких rate limits, но рекомендуется ≤50 RPS
    - Высокое качество данных с ручной модерацией
    """
    
    BASE_URL = "https://api.ror.org"
    
    # Маппинг типов ROR -> наши типы
    TYPE_MAP = {
        "education": OrganizationType.UNIVERSITY,
        "company": OrganizationType.COMPANY,
        "government": OrganizationType.GOVERNMENT,
        "nonprofit": OrganizationType.NONPROFIT,
        "healthcare": OrganizationType.HOSPITAL,
        "facility": OrganizationType.RESEARCH_INSTITUTE,
        "archive": OrganizationType.RESEARCH_INSTITUTE,
        "other": OrganizationType.UNKNOWN
    }
    
    # Словарь известных аббревиатур -> полное название
    ABBREVIATION_MAP = {
        "mit": "Massachusetts Institute of Technology",
        "caltech": "California Institute of Technology",
        "cmu": "Carnegie Mellon University",
        "eth": "ETH Zurich",
        "eth zurich": "ETH Zurich",
        "ucl": "University College London",
        "ucla": "University of California, Los Angeles",
        "ucb": "University of California, Berkeley",
        "uc berkeley": "University of California, Berkeley",
        "usc": "University of Southern California",
        "nyu": "New York University",
        "columbia": "Columbia University",
        "princeton": "Princeton University",
        "yale": "Yale University",
        "penn": "University of Pennsylvania",
        "upenn": "University of Pennsylvania",
        "gatech": "Georgia Institute of Technology",
        "georgia tech": "Georgia Institute of Technology",
        "uiuc": "University of Illinois Urbana-Champaign",
        "umich": "University of Michigan",
        "uw": "University of Washington",
        "ut austin": "University of Texas at Austin",
        "utexas": "University of Texas at Austin",
        "google research": "Google LLC",
        "google deepmind": "Google DeepMind",
        "deepmind": "Google DeepMind",
        "meta ai": "Meta Platforms",
        "facebook ai": "Meta Platforms",
        "fair": "Meta Platforms",
        "microsoft research": "Microsoft",
        "msr": "Microsoft",
        "openai": "OpenAI",
        "amazon research": "Amazon.com",
        "apple ml": "Apple Inc.",
        "ibm research": "IBM",
        "nvidia research": "NVIDIA",
        "inria": "Institut national de recherche en sciences et technologies du numérique",
        "cnrs": "Centre National de la Recherche Scientifique",
        "max planck": "Max Planck Society",
        "mpi": "Max Planck Society",
        "csiro": "Commonwealth Scientific and Industrial Research Organisation",
        "nist": "National Institute of Standards and Technology",
        "nasa": "National Aeronautics and Space Administration",
        "nih": "National Institutes of Health",
        "epfl": "École Polytechnique Fédérale de Lausanne",
        "kaist": "Korea Advanced Institute of Science and Technology",
        "postech": "Pohang University of Science and Technology",
        "nus": "National University of Singapore",
        "ntu": "Nanyang Technological University",
        "hku": "University of Hong Kong",
        "cuhk": "Chinese University of Hong Kong",
        "pku": "Peking University",
        "thu": "Tsinghua University",
        "sjtu": "Shanghai Jiao Tong University",
        "zju": "Zhejiang University",
        "ustc": "University of Science and Technology of China",
        "anu": "Australian National University",
        "uoft": "University of Toronto",
        "mcgill": "McGill University",
        "ubc": "University of British Columbia",
        "cam": "University of Cambridge",
        "ox": "University of Oxford",
        "oxford": "University of Oxford",
        "cambridge": "University of Cambridge",
        "imperial": "Imperial College London",
        "ic": "Imperial College London",
        "lmu": "Ludwig-Maximilians-Universität München",
        "tu munich": "Technische Universität München",
        "tum": "Technische Universität München"
    }
    
    def __init__(self, cache_size: int = 1000):
        """
        Args:
            cache_size: Максимальный размер кэша результатов
        """
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={"Accept": "application/json"},
            timeout=30.0
        )
        
        self._cache: Dict[str, Dict] = {}
        self._cache_size = cache_size
        self._last_request_time = 0
        self._min_request_interval = 0.05  # ~20 RPS max
        self._request_count = 0
    
    def _rate_limit(self):
        """Соблюдение rate limit"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=5)
    )
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Выполнить запрос к API"""
        self._rate_limit()
        self._request_count += 1
        
        response = self._client.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    
    def lookup(self, org_name: str) -> Optional[Dict[str, Any]]:
        """
        Поиск организации по названию.
        
        Args:
            org_name: Название организации
            
        Returns:
            Данные организации или None
        """
        # Проверка кэша
        cache_key = org_name.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Проверяем словарь аббревиатур
        search_name = org_name
        if cache_key in self.ABBREVIATION_MAP:
            search_name = self.ABBREVIATION_MAP[cache_key]
        
        result = self._search(search_name)
        
        # Сохранение в кэш
        if len(self._cache) >= self._cache_size:
            # Удаляем старейший элемент (простой LRU)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = result
        
        return result
    
    def _search(self, org_name: str) -> Optional[Dict[str, Any]]:
        """Выполнить поиск в ROR API"""
        try:
            data = self._make_request("/organizations", {"query": org_name})
        except Exception as e:
            print(f"[ROR] Search error for '{org_name}': {e}")
            return None
        
        items = data.get("items", [])
        if not items:
            return None
        
        # Выбираем лучшее совпадение
        best_match = None
        best_score = 0
        
        for item in items[:5]:  # Проверяем топ-5 результатов
            names_to_check = []
            
            # ROR API v2: названия в поле 'names' (массив объектов)
            for name_obj in item.get("names", []):
                name_value = name_obj.get("value", "")
                if name_value:
                    names_to_check.append(name_value)
            
            # Ищем лучшее совпадение по названию
            for name in names_to_check:
                score = fuzz.token_sort_ratio(org_name.lower(), name.lower())
                if score > best_score:
                    best_score = score
                    best_match = item
        
        # Требуем минимум 60% совпадения (снижено для аббревиатур)
        if best_score < 60:
            return None
        
        return self._convert_result(best_match, best_score / 100.0)
    
    def get_by_id(self, ror_id: str) -> Optional[Dict[str, Any]]:
        """
        Получить организацию по ROR ID.
        
        Args:
            ror_id: ROR ID (например, "https://ror.org/0168r3w48")
            
        Returns:
            Данные организации или None
        """
        # Извлекаем ID из полного URL если нужно
        if ror_id.startswith("https://ror.org/"):
            ror_id = ror_id.replace("https://ror.org/", "")
        
        try:
            data = self._make_request(f"/organizations/{ror_id}")
            return self._convert_result(data, 1.0)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except Exception as e:
            print(f"[ROR] Error fetching {ror_id}: {e}")
            return None
    
    def _convert_result(self, data: Dict, confidence: float) -> Dict[str, Any]:
        """Конвертация результата ROR API v2 в унифицированный формат"""
        # ROR API v2: страна в locations[0].geonames_details
        country_name = "Unknown"
        country_code = "XX"
        locations = data.get("locations", [])
        if locations:
            geonames = locations[0].get("geonames_details", {})
            country_name = geonames.get("country_name", "Unknown")
            country_code = geonames.get("country_code", "XX")
        
        # Определяем тип организации
        types = data.get("types", [])
        org_type = OrganizationType.UNKNOWN
        for t in types:
            if t.lower() in self.TYPE_MAP:
                org_type = self.TYPE_MAP[t.lower()]
                break
        
        # ROR API v2: основное имя из names с типом ror_display
        canonical_name = ""
        aliases = []
        for name_obj in data.get("names", []):
            name_types = name_obj.get("types", [])
            name_value = name_obj.get("value", "")
            if "ror_display" in name_types:
                canonical_name = name_value
            elif "alias" in name_types or "acronym" in name_types:
                aliases.append(name_value)
        
        # Извлекаем ссылки
        links = []
        wikipedia_url = None
        for link in data.get("links", []):
            link_type = link.get("type", "")
            link_value = link.get("value", "")
            if link_type == "wikipedia":
                wikipedia_url = link_value
            else:
                links.append(link_value)
        
        return {
            "ror_id": data.get("id"),
            "name": canonical_name,
            "country": country_name,
            "country_code": country_code,
            "type": org_type,
            "aliases": aliases,
            "links": links,
            "wikipedia_url": wikipedia_url,
            "confidence": confidence
        }
    
    def get_request_count(self) -> int:
        """Количество выполненных запросов"""
        return self._request_count
    
    def clear_cache(self):
        """Очистить кэш"""
        self._cache.clear()
    
    def close(self):
        """Закрыть HTTP клиент"""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# Глобальный экземпляр для переиспользования
_ror_instance: Optional[RORLookup] = None


def get_ror_lookup() -> RORLookup:
    """Получить глобальный экземпляр ROR lookup"""
    global _ror_instance
    if _ror_instance is None:
        _ror_instance = RORLookup()
    return _ror_instance


def lookup_ror(org_name: str) -> Optional[Dict[str, Any]]:
    """
    Удобная функция для поиска организации в ROR.
    
    Args:
        org_name: Название организации
        
    Returns:
        Данные организации или None
    """
    ror = get_ror_lookup()
    return ror.lookup(org_name)
