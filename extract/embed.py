# you have to pass a python dictionary
# if you wanna pass text I can modify code

import re
from statistics import mean
import unicodedata
import numpy as np

# import pandas as pd

from typing import Any, Callable, List, Tuple, TypedDict, cast
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer  # type: ignore
from sentence_transformers.util import cos_sim  # type: ignore

from nltk.tokenize import word_tokenize  # type:ignore
from nltk import tokenize  # type:ignore

from extract.cache import skill_framework_cache

# from prisma.models import SkillCluster  # type: ignore
from extract.info import EmbedSource, Embeddings
from extract.config import Config

# calculate similarity
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def clean_text(text: str):
    soup = BeautifulSoup(text, features="lxml")
    a = soup.get_text()
    x = re.sub("\n", " ", a)
    x = re.sub("\\\\n", " ", x)
    y = re.sub("\t", " ", x)
    z = re.sub("\\S+@\\S+", " ", y)
    w = re.sub("http[s]?\\://\\S+", " ", z)
    q = unicodedata.normalize("NFKD", w)
    u = re.sub(r"(\xe9|\362)", "", q)
    return u


def split_large_sentence(text: str):
    text_tokens: List[str] = word_tokenize(text)
    result = [text_tokens[i : i + 128] for i in range(0, len(text_tokens), 128)]

    return result


def join_chunks(current_chunks: List[str]):
    text = (" ").join(current_chunks)
    text = re.sub(r" ([\.,;'\":\?\!])", r"\g<1>", text)
    return text


def create_chunks(text: str, chunk_size: int = 128):
    sentences: List[str] = tokenize.sent_tokenize(clean_text(text))
    combined: List[str] = []
    # current = ""
    current_chunks = []
    for sentence in sentences:
        chunks = split_large_sentence(sentence)

        if len(chunks) > 1:
            for chunk in chunks:
                if len(current_chunks) + len(chunk) > chunk_size:
                    combined.append(join_chunks(current_chunks))
                    current_chunks = chunk
                else:
                    current_chunks.extend(chunk)
        else:
            if len(current_chunks) + len(chunks[0]) > chunk_size:
                combined.append(join_chunks(current_chunks))
                current_chunks = chunks[0]
            else:
                current_chunks.extend(chunks[0])

    if len(current_chunks) > 0:
        combined.append(join_chunks(current_chunks))

    return combined


def get_embeddings(text: str, with_text: bool = False) -> Tuple[Embeddings, List[str]]:
    """
    Split texts into sentences and get embeddings for each sentence.
    The final embeddings is the mean of all sentence embeddings.
    :param text: str. Input text.
    :return: np.array. Embeddings.
    """

    combined = create_chunks(text, chunk_size=128)
    result: Embeddings = cast(Embeddings, [model.encode(x) for x in combined])

    return (result, combined)


def get_embeddings_fast(text: str) -> List[Any]:
    embeddings, _ = get_embeddings(text)
    return np.mean(embeddings, axis=0)


class EmbedResult(TypedDict):
    id: int
    name: str
    max: float
    description: float
    title: float
    keywords: List[str]
    keyword_count: int
    values: List[Any]


multipliers = {0: 1, 1: 1.4, 2: 1.5, 3: 1.6, 4: 1.7}


def keyword_multiplier(count: int) -> float:
    return multipliers[count] if count in multipliers else 2


def combine(title_rank: float, description_rank: float | None, keyword_count: int) -> float:
    if description_rank is None:
        return 0 if keyword_count == 0 else np.clip(0.5 * keyword_multiplier(keyword_count), 0, 1)
    return np.clip((0.4 * description_rank + 0.6 * title_rank) * keyword_multiplier(keyword_count), 0, 1)


class SimilarityResult(TypedDict):
    value: float
    source: str
    target: str


def compare_embeddings(
    in_embeddings: Embeddings, in_texts: List[str], for_embeddings: Embeddings, for_texts: List[str]
):
    combinations: List[SimilarityResult] = []

    for i in range(len(in_embeddings)):
        for_combinations: List[SimilarityResult] = []
        for j in range(len(for_embeddings)):
            for_combinations.append(
                {
                    "value": cos_sim(in_embeddings[i], for_embeddings[j]).numpy()[0][0],
                    "source": in_texts[i],
                    "target": for_texts[j],
                }
            )
        if len(for_combinations) > 0:
            description_embedding = max(for_combinations, key=lambda x: x["value"])
            combinations.append(description_embedding)

    if len(combinations) > 0:
        # this one calculates the
        return mean([x["value"] for x in combinations])
    return 0


def calculate_complex_similarity(
    in_this: List[EmbedSource],
    for_this: Embeddings,
    texts: List[str],
    text: str,
    threshold: float | None = 0.5,
    checker: Callable[[int, float, List[EmbedResult]], bool] | None = None,
    allow_multipliers: bool = True,
) -> List[EmbedResult]:

    in_scores: List[EmbedResult] = []

    # find the best
    for in_value in in_this:
        if in_value["keywords_reg"] is not None:
            keyword_matches = in_value["keywords_reg"].findall(text)
        else:
            keyword_matches = []

        if len(in_value["description_chunks"]) > 0:
            description_embedding = compare_embeddings(
                in_value["description_embeddings"], in_value["description_chunks"], for_this, texts
            )
        else:
            description_embedding = None
        name_embedding = compare_embeddings(in_value["name_embeddings"], in_value["name_chunks"], for_this, texts)
        combined = combine(
            name_embedding, description_embedding, 0
        )  # len(keyword_matches))  # ) len(in_value["keywords"]))

        if (threshold is not None and combined > threshold) or (
            checker is not None and checker(in_value["id"], combined, in_scores)
        ):
            in_scores.append(
                {
                    "id": in_value["id"],
                    "name": in_value["name"],
                    "max": combined * (in_value["multiplier"] if "multiplier" in in_value and allow_multipliers else 1),
                    "description": description_embedding or 0,
                    "title": name_embedding,
                    "keyword_count": len(keyword_matches),
                    "keywords": keyword_matches,
                    "values": [],
                }
            )

    return sorted(in_scores, reverse=True, key=lambda x: x["max"])


def calculate_cluster_similarity(
    text: str, threshold: float = Config.cluster_match_threshold, allow_multipliers: bool = True
) -> List[EmbedResult]:
    for_this, texts = get_embeddings(text, with_text=True)

    return calculate_complex_similarity(
        get_cluster_cache(), for_this, texts, text, threshold=threshold, allow_multipliers=allow_multipliers
    )


def calculate_similarity_multi(in_this: Any, text: str) -> Any:

    for_this, texts = get_embeddings(text, with_text=True)
    in_scores: List[Any] = []

    # find the best
    for in_value in in_this:

        description_embedding = compare_embeddings(in_value["embeddings"], in_value["texts"], for_this, texts)

        if description_embedding > 0.5:
            in_scores.append(
                {
                    "id": in_value["id"],
                    "name": in_value["name"],
                    "max": description_embedding,
                }
            )

    # print(lg_sfia_score)
    return sorted(in_scores, reverse=True, key=lambda x: x["max"])


def calculate_similarity_fast(in_what: List[EmbedSource], for_what: Any) -> List[EmbedResult]:

    in_scores: List[EmbedResult] = []

    for in_value in in_what:
        if len(in_value["mean"]) > 0:
            in_scores.append(
                {
                    "id": in_value["id"],
                    "name": in_value["name"],
                    "max": cos_sim(in_value["mean"], for_what).numpy()[0][0],  # type: ignore
                    "values": [],
                }
            )

    # print(lg_sfia_score)
    return sorted(in_scores, reverse=True, key=lambda x: x["max"])


cluster_cache: List[EmbedSource] | None = None


def get_cluster_cache():
    global cluster_cache
    if cluster_cache is None:
        cluster_cache = skill_framework_cache.skills.get(Config.skill_framework)
        if cluster_cache is None:
            raise RuntimeError("You must pre-cache the clusters!")
    return cluster_cache


def calculate_cluster_similarity_fast(text: str):
    return calculate_similarity_fast(get_cluster_cache(), get_embeddings_fast(text))


def calculate_similarity(in_what: str, for_what: str) -> float:

    cleaned_in = clean_text(in_what)
    cleaned_for = clean_text(for_what)

    # getting embeddings
    embed_in: Any = model.encode(cleaned_in)
    embed_for: Any = model.encode(cleaned_for)

    similarity = cos_sim(embed_in, embed_for).numpy()[0][0]

    return similarity
