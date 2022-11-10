from __future__ import annotations
import re

from typing import Any, List, Literal, Optional, TypedDict
from torch import Tensor

import pandas as pd

SkillFramework = Literal["Bg"] | Literal["Asf"]
Embeddings = List[Tensor]

class Em:
    def __init__(self, id: int, name: str, description: str, keywords: List[str], multiplier: float = 1.0):
        self.id = id
        self.name = name
        self.description = description
        self.keywords = keywords
        self.multiplier = multiplier


class MatchReference(TypedDict):
    id: str
    name: str
    skill_match: float
    skills: "pd.Series[float]"
    match: List[float]  # [45% covered, 0.92 level]


class EmbedSource(TypedDict):
    id: int
    name: str
    description: str
    name_embeddings: List[Any]
    name_chunks: List[str]
    multiplier: float
    description_embeddings: List[Any]
    description_chunks: List[str]
    keywords: List[str] | None
    keywords_reg: re.Pattern[str] | None
    mean: Any


class SkillCluster(TypedDict):
    categories: Optional[List[int]]
    description: Optional[str]
    id: int
    name: str


class Skill(TypedDict):
    clusters: Optional[List[int]]
    description: Optional[str]
    id: int
    keywords: Optional[List[int]]
    name: str
    uid: Optional[int]


class SkillCategory(TypedDict):
    description: Optional[str]
    id: int
    name: str
