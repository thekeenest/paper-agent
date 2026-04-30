"""
Модуль аналитики и визуализации результатов.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class AnalyticsEngine:
    """
    Движок аналитики для обработки результатов извлечения.
    """
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df: Optional[pd.DataFrame] = None
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Загрузка данных из CSV"""
        self.df = pd.read_csv(csv_path)
        return self.df
    
    def load_from_papers(self, papers: List[Any]) -> pd.DataFrame:
        """Загрузка данных из списка PaperMetadata"""
        records = []
        for paper in papers:
            for author in paper.authors:
                records.append({
                    "paper_id": paper.arxiv_id,
                    "paper_title": paper.title,
                    "published_date": paper.published_date,
                    "categories": ",".join(paper.categories) if paper.categories else "",
                    "author_name": author.name,
                    "raw_affiliation": author.raw_affiliation,
                    "normalized_affiliation": author.normalized_affiliation,
                    "country": author.country,
                    "country_code": author.country_code,
                    "org_type": author.org_type.value if author.org_type else "unknown",
                    "confidence": author.confidence
                })
        
        self.df = pd.DataFrame(records)
        return self.df
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Получить сводную статистику"""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        return {
            "total_papers": self.df["paper_id"].nunique(),
            "total_authors": len(self.df),
            "unique_authors": self.df["author_name"].nunique(),
            "unique_organizations": self.df["normalized_affiliation"].nunique(),
            "unique_countries": self.df["country"].nunique(),
            "avg_authors_per_paper": len(self.df) / self.df["paper_id"].nunique(),
            "avg_confidence": self.df["confidence"].mean()
        }
    
    def get_top_organizations(self, n: int = 20) -> pd.DataFrame:
        """Топ организаций по числу авторов"""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        return (
            self.df
            .groupby("normalized_affiliation")
            .agg({
                "author_name": "count",
                "country": "first",
                "org_type": "first"
            })
            .rename(columns={"author_name": "author_count"})
            .sort_values("author_count", ascending=False)
            .head(n)
            .reset_index()
        )
    
    def get_country_distribution(self) -> pd.DataFrame:
        """Распределение по странам"""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        return (
            self.df
            .groupby("country")
            .agg({
                "author_name": "count",
                "normalized_affiliation": "nunique"
            })
            .rename(columns={
                "author_name": "author_count",
                "normalized_affiliation": "org_count"
            })
            .sort_values("author_count", ascending=False)
            .reset_index()
        )
    
    def get_org_type_distribution(self) -> pd.DataFrame:
        """Распределение по типам организаций"""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        return (
            self.df
            .groupby("org_type")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    
    # ============================================================
    # ВИЗУАЛИЗАЦИИ
    # ============================================================
    
    def plot_top_organizations(
        self, 
        n: int = 15, 
        figsize: tuple = (12, 8),
        save: bool = True
    ) -> plt.Figure:
        """
        График топ организаций.
        """
        top_orgs = self.get_top_organizations(n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Цвета по типу организации
        colors = {
            "company": "#FF6B6B",
            "university": "#4ECDC4",
            "research_institute": "#45B7D1",
            "unknown": "#95A5A6"
        }
        bar_colors = [colors.get(t, colors["unknown"]) for t in top_orgs["org_type"]]
        
        bars = ax.barh(
            range(len(top_orgs)),
            top_orgs["author_count"],
            color=bar_colors
        )
        
        ax.set_yticks(range(len(top_orgs)))
        ax.set_yticklabels(top_orgs["normalized_affiliation"])
        ax.invert_yaxis()
        
        ax.set_xlabel("Number of Authors", fontsize=12)
        ax.set_title(f"Top {n} Organizations by Author Count", fontsize=14, fontweight="bold")
        
        # Легенда
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors["company"], label="Company"),
            Patch(facecolor=colors["university"], label="University"),
            Patch(facecolor=colors["research_institute"], label="Research Institute"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")
        
        # Значения на барах
        for bar, count in zip(bars, top_orgs["author_count"]):
            ax.text(
                bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(count), va="center", fontsize=10
            )
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "top_organizations.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        
        return fig
    
    def plot_country_distribution(
        self,
        n: int = 10,
        figsize: tuple = (10, 10),
        save: bool = True
    ) -> plt.Figure:
        """
        Круговая диаграмма распределения по странам.
        """
        countries = self.get_country_distribution().head(n)
        
        # Группируем мелкие страны в "Other"
        total = countries["author_count"].sum()
        threshold = total * 0.02  # 2%
        
        main_countries = countries[countries["author_count"] >= threshold]
        other_count = countries[countries["author_count"] < threshold]["author_count"].sum()
        
        if other_count > 0:
            other_row = pd.DataFrame([{"country": "Other", "author_count": other_count}])
            main_countries = pd.concat([main_countries, other_row], ignore_index=True)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        wedges, texts, autotexts = ax.pie(
            main_countries["author_count"],
            labels=main_countries["country"],
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("husl", len(main_countries))
        )
        
        ax.set_title("Author Distribution by Country", fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "country_distribution.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        
        return fig
    
    def plot_org_type_distribution(
        self,
        figsize: tuple = (8, 6),
        save: bool = True
    ) -> plt.Figure:
        """
        Распределение по типам организаций.
        """
        org_types = self.get_org_type_distribution()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = {
            "company": "#FF6B6B",
            "university": "#4ECDC4",
            "research_institute": "#45B7D1",
            "government": "#9B59B6",
            "unknown": "#95A5A6"
        }
        bar_colors = [colors.get(t, colors["unknown"]) for t in org_types["org_type"]]
        
        ax.bar(org_types["org_type"], org_types["count"], color=bar_colors)
        
        ax.set_xlabel("Organization Type", fontsize=12)
        ax.set_ylabel("Number of Authors", fontsize=12)
        ax.set_title("Distribution by Organization Type", fontsize=14, fontweight="bold")
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "org_type_distribution.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        
        return fig
    
    def plot_industry_vs_academia(
        self,
        figsize: tuple = (10, 6),
        save: bool = True
    ) -> plt.Figure:
        """
        Сравнение индустрии и академии.
        """
        if self.df is None:
            raise ValueError("Data not loaded")
        
        # Классификация
        industry_types = ["company"]
        academia_types = ["university", "research_institute"]
        
        industry_count = self.df[self.df["org_type"].isin(industry_types)].shape[0]
        academia_count = self.df[self.df["org_type"].isin(academia_types)].shape[0]
        other_count = self.df[~self.df["org_type"].isin(industry_types + academia_types)].shape[0]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        categories = ["Industry", "Academia", "Other/Unknown"]
        counts = [industry_count, academia_count, other_count]
        colors = ["#FF6B6B", "#4ECDC4", "#95A5A6"]
        
        bars = ax.bar(categories, counts, color=colors)
        
        # Проценты
        total = sum(counts)
        for bar, count in zip(bars, counts):
            percentage = count / total * 100
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{count}\n({percentage:.1f}%)",
                ha="center", va="bottom", fontsize=11
            )
        
        ax.set_ylabel("Number of Authors", fontsize=12)
        ax.set_title("Industry vs Academia", fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "industry_vs_academia.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        
        return fig
    
    def generate_all_plots(self) -> List[str]:
        """
        Генерация всех графиков.
        
        Returns:
            Список путей к сохранённым файлам
        """
        paths = []
        
        try:
            self.plot_top_organizations()
            paths.append(str(self.output_dir / "top_organizations.png"))
        except Exception as e:
            print(f"Error plotting top_organizations: {e}")
        
        try:
            self.plot_country_distribution()
            paths.append(str(self.output_dir / "country_distribution.png"))
        except Exception as e:
            print(f"Error plotting country_distribution: {e}")
        
        try:
            self.plot_org_type_distribution()
            paths.append(str(self.output_dir / "org_type_distribution.png"))
        except Exception as e:
            print(f"Error plotting org_type_distribution: {e}")
        
        try:
            self.plot_industry_vs_academia()
            paths.append(str(self.output_dir / "industry_vs_academia.png"))
        except Exception as e:
            print(f"Error plotting industry_vs_academia: {e}")
        
        return paths
    
    def export_to_latex_table(self, top_n: int = 10) -> str:
        """
        Экспорт топ организаций в LaTeX таблицу.
        """
        top_orgs = self.get_top_organizations(top_n)
        
        latex = r"""
\begin{table}[H]
\centering
\caption{Top %d Organizations by Author Count}
\label{tab:top_orgs}
\begin{tabular}{clccc}
\toprule
№ & Organization & Authors & Country & Type \\
\midrule
""" % top_n
        
        for i, row in enumerate(top_orgs.itertuples(), 1):
            latex += f"{i} & {row.normalized_affiliation} & {row.author_count} & {row.country} & {row.org_type} \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return latex


def analyze_results(csv_path: str, output_dir: str = "./output"):
    """
    Утилита для быстрого анализа результатов.
    
    Args:
        csv_path: Путь к CSV с аффилиациями
        output_dir: Директория для сохранения графиков
    """
    engine = AnalyticsEngine(output_dir)
    engine.load_data(csv_path)
    
    print("\n" + "="*50)
    print("ANALYTICS SUMMARY")
    print("="*50)
    
    stats = engine.get_summary_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "-"*50)
    print("TOP 10 ORGANIZATIONS:")
    print("-"*50)
    top_orgs = engine.get_top_organizations(10)
    for i, row in enumerate(top_orgs.itertuples(), 1):
        print(f"  {i}. {row.normalized_affiliation}: {row.author_count} authors ({row.country})")
    
    print("\n" + "-"*50)
    print("GENERATING PLOTS...")
    print("-"*50)
    
    paths = engine.generate_all_plots()
    for path in paths:
        print(f"  ✓ {path}")
    
    print("\n" + "="*50)
    print("Analysis complete!")
    print("="*50)
    
    return engine
