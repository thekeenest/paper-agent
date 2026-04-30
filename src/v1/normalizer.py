"""
Нормализатор названий организаций.

Использует базу знаний + fuzzy matching + LLM fallback.
"""

import os
from typing import Optional, Dict, Any, List
from rapidfuzz import fuzz, process
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .models import OrganizationType, NormalizationResult
from .knowledge_base import ORGANIZATION_KB, lookup_organization, VARIANT_LOOKUP


class OrganizationNormalizer:
    """
    Нормализатор названий организаций.
    
    Стратегия:
    1. Точный поиск в Knowledge Base
    2. Fuzzy matching для похожих названий
    3. LLM fallback для неизвестных организаций
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        fuzzy_threshold: int = 80,
        use_llm_fallback: bool = True
    ):
        """
        Args:
            llm_model: Модель для LLM fallback
            fuzzy_threshold: Порог схожести для fuzzy matching (0-100)
            use_llm_fallback: Использовать ли LLM для неизвестных организаций
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.use_llm_fallback = use_llm_fallback
        
        # Подготовка списка для fuzzy matching
        self._all_variants = list(VARIANT_LOOKUP.keys())
        
        # LLM для fallback
        if use_llm_fallback:
            self._llm = ChatOpenAI(
                model=llm_model,
                temperature=0
            )
        else:
            self._llm = None
        
        # Кэш результатов нормализации
        self._cache: Dict[str, NormalizationResult] = {}
    
    def normalize(self, raw_affiliation: str) -> NormalizationResult:
        """
        Нормализовать название организации.
        
        Args:
            raw_affiliation: Исходное название организации
        
        Returns:
            NormalizationResult с нормализованными данными
        """
        # Проверка кэша
        cache_key = raw_affiliation.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = self._normalize_internal(raw_affiliation)
        self._cache[cache_key] = result
        return result
    
    def _normalize_internal(self, raw: str) -> NormalizationResult:
        """Внутренняя логика нормализации"""
        
        # Шаг 1: Точный поиск в KB
        kb_result = lookup_organization(raw)
        if kb_result:
            return NormalizationResult(
                original=raw,
                normalized=kb_result["canonical"],
                country=kb_result["country"],
                country_code=kb_result["country_code"],
                org_type=OrganizationType(kb_result["type"]),
                confidence=0.95,
                source="kb"
            )
        
        # Шаг 2: Fuzzy matching
        fuzzy_result = self._fuzzy_match(raw)
        if fuzzy_result:
            return fuzzy_result
        
        # Шаг 3: LLM fallback
        if self.use_llm_fallback and self._llm:
            llm_result = self._llm_normalize(raw)
            if llm_result:
                return llm_result
        
        # Fallback: вернуть как есть
        return NormalizationResult(
            original=raw,
            normalized=raw,
            country="Unknown",
            country_code="XX",
            org_type=OrganizationType.UNKNOWN,
            confidence=0.3,
            source="none"
        )
    
    def _fuzzy_match(self, raw: str) -> Optional[NormalizationResult]:
        """Fuzzy matching по базе знаний"""
        raw_lower = raw.lower().strip()
        
        # Найти лучшее совпадение
        match = process.extractOne(
            raw_lower,
            self._all_variants,
            scorer=fuzz.token_sort_ratio
        )
        
        if match and match[1] >= self.fuzzy_threshold:
            matched_variant = match[0]
            kb_key = VARIANT_LOOKUP[matched_variant]
            kb_data = ORGANIZATION_KB[kb_key]
            
            # Конвертировать score в confidence (0.0-1.0)
            confidence = match[1] / 100.0 * 0.9  # Max 0.9 для fuzzy
            
            return NormalizationResult(
                original=raw,
                normalized=kb_data["canonical"],
                country=kb_data["country"],
                country_code=kb_data["country_code"],
                org_type=OrganizationType(kb_data["type"]),
                confidence=confidence,
                source="fuzzy"
            )
        
        return None
    
    def _llm_normalize(self, raw: str) -> Optional[NormalizationResult]:
        """LLM-based нормализация для неизвестных организаций"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in academic institutions and tech companies.
Given an organization name, provide:
1. The canonical (official) name
2. The country where it is headquartered
3. The type: university, company, research_institute, government, hospital, nonprofit

Respond in JSON format with fields: canonical, country, country_code (ISO 3166-1 alpha-2), type"""),
            ("user", "Organization: {org}")
        ])
        
        try:
            from langchain_core.output_parsers import JsonOutputParser
            
            chain = prompt | self._llm | JsonOutputParser()
            result = chain.invoke({"org": raw})
            
            # Валидация типа
            org_type_str = result.get("type", "unknown")
            try:
                org_type = OrganizationType(org_type_str)
            except ValueError:
                org_type = OrganizationType.UNKNOWN
            
            return NormalizationResult(
                original=raw,
                normalized=result.get("canonical", raw),
                country=result.get("country", "Unknown"),
                country_code=result.get("country_code", "XX"),
                org_type=org_type,
                confidence=0.7,
                source="llm"
            )
        except Exception as e:
            print(f"LLM normalization failed for '{raw}': {e}")
            return None
    
    def normalize_batch(self, affiliations: List[str]) -> List[NormalizationResult]:
        """
        Нормализовать список аффилиаций.
        
        Args:
            affiliations: Список названий организаций
        
        Returns:
            Список NormalizationResult
        """
        return [self.normalize(aff) for aff in affiliations]
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику нормализации"""
        if not self._cache:
            return {"total": 0}
        
        sources = {}
        for result in self._cache.values():
            sources[result.source] = sources.get(result.source, 0) + 1
        
        return {
            "total": len(self._cache),
            "by_source": sources,
            "avg_confidence": sum(r.confidence for r in self._cache.values()) / len(self._cache)
        }


# Глобальный экземпляр для удобства использования
_normalizer: Optional[OrganizationNormalizer] = None


def get_normalizer() -> OrganizationNormalizer:
    """Получить глобальный экземпляр нормализатора"""
    global _normalizer
    if _normalizer is None:
        _normalizer = OrganizationNormalizer()
    return _normalizer


def normalize_affiliation(raw: str) -> NormalizationResult:
    """Утилита для быстрой нормализации"""
    return get_normalizer().normalize(raw)
