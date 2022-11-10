from typing import Dict, List, Tuple
from extract.config import Config
from extract.cache import skill_framework_cache
from extract.info import SkillFramework

from frameworks.asf.load_framework import load_asf_framework
from frameworks.bg.load_framework import load_bg_framework

from .embed import EmbedSource, calculate_complex_similarity, get_embeddings  # , get_embeddings_fast
from .progress import ProgressBar

import nltk  # type: ignore
import numpy as np
import re

# download the tokenizer source


# async def pre_cache_anzsco_embeddings():
#     prisma = await connect()
#     anzsco = await prisma.anzsco.find_many()

#     description_cache = []
#     progress = ProgressBar()
#     progress.start_sequence(len(anzsco), "Processing codes")

#     for item in anzsco:
#         (embeddings, texts) = get_embeddings((item.name or "") + "\n" + (item.description or ""), with_text=True)
#         description_cache.append({"id": item.id, "name": item.name, "embeddings": embeddings, "texts": texts})
#         progress.step(f"Processed: {item.id}")

#     persistent_cache.descriptions.store("descriptions", description_cache)


# async def read_roles() -> List[Em]:
#     prisma = await connect()
#     anzsco = await prisma.anzsco.find_many()
#     return [Em(x.id, x.name or "", x.description or "", []) for x in anzsco]


# async def pre_cache_anzsco_embeddings():

#     anzsco = await read_roles()

#     description_cache: List[EmbedSource] = []
#     progress = ProgressBar()
#     progress.start_sequence(len(anzsco), "Processing codes")

#     for item in anzsco:
#         (de, dt) = get_embeddings(item.description or "", with_text=True)
#         (te, tt) = get_embeddings(item.name or "", with_text=True)

#         name = item.name or ""

#         description_cache.append(
#             {
#                 "id": item.id,
#                 "name": item.name or "",
#                 "description": item.description or "",
#                 "mean": 0,
#                 "description_embeddings": de,
#                 "description_chunks": dt,
#                 "name_embeddings": te,
#                 "multiplier": 1,
#                 "name_chunks": tt,
#                 "keywords": [name],
#                 "keywords_reg": re.compile(r"(" + name + ")"),
#             }
#         )
#         progress.step(f"Processed: {item.id}")

#     # df = pd.DataFrame(description_cache, index=[x.id for x in anzsco], columns=["embeddings"])
#     persistent_cache.descriptions.set(description_cache, "descriptions")
#     # save_to_cache("embeddings/anzsco_df.pkl", df)


# async def pre_cache_anzsco_embeddings_fast():
#     prisma = await connect()
#     anzsco = await prisma.anzsco.find_many()

#     description_cache = []
#     progress = ProgressBar()
#     progress.start_sequence(len(anzsco), "Processing codes")

#     for item in anzsco:
#         description_cache.append(
#             {
#                 "id": item.id,
#                 "name": item.name,
#                 "embeddings": get_embeddings_fast((item.name or "") + "\n" + (item.description or "")),
#             }
#         )
#         progress.step(f"Processed: {item.id}")

#     # df = pd.DataFrame(description_cache, index=[x.id for x in anzsco], columns=["embeddings"])
#     persistent_cache.descriptions.store("descriptions", description_cache)
#     # save_to_cache("embeddings/anzsco_df.pkl", df)


def createSkillRegExp(skill: str):
    if len(skill) < 3:
        return "(?<![\\.-])\\b" + skill + "\\b"
    else:
        return "\\b" + skill + "\\b"


# async def read_bg_skills() -> List[Em]:
#     prisma = await connect()
#     clusters = await prisma.skillcluster.find_many(
#         include={"Descriptions": True, "Skills": {"include": {"Skill": True}}}
#     )
#     return [
#         Em(
#             x.id,
#             x.name or "",
#             x.Descriptions[0].description if x.Descriptions and len(x.Descriptions) > 0 else "",
#             [re.escape(y.Skill.name) for y in x.Skills if y.Skill] if x.Skills else [],
#         )
#         for x in clusters
#     ]


async def cache_skill_embeddings(framework: SkillFramework = "Asf"):

    nltk.download("punkt")

    # good_clusters = await read_bg_skills()

    if framework == "Asf":
        good_clusters = load_asf_framework()
    else:
        good_clusters = load_bg_framework()

    # good_clusters = [x for x in clusters if x.descriptions is not None and len(x.descriptions) > 0]
    cluster_list: List[EmbedSource] = []

    progress = ProgressBar(new_line=True)
    progress.start_sequence(len(good_clusters), "Step 2: Create cluster embeddings")

    for cluster in good_clusters:

        # if cluster.id == 1:
        #     continue

        progress.step(str(cluster.name).ljust(30, " ")[0:30] + ("..." if len(cluster.name) > 30 else "   "))

        description: str = ""

        # if cluster.descriptions is not None and len(cluster.descriptions) > 0:

        if cluster.description != "":
            description = cluster.description
            (description_embeddings, description_texts) = get_embeddings(description, with_text=True)
        else:
            description_embeddings = []
            description_texts = []
        (title_embeddings, title_texts) = get_embeddings(cluster.name, with_text=True)

        if len(cluster.keywords) > 0:
            keyword_res = [createSkillRegExp(x) for x in cluster.keywords]  # type: ignore
            keyword_joined = "(" + "|".join(keyword_res) + ")"
            keyword_re = re.compile(keyword_joined, re.IGNORECASE)
        else:
            keyword_res = None
            keyword_re = None

        cluster_list.append(
            {
                "id": cluster.id,
                "name": cluster.name,
                "name_embeddings": title_embeddings,
                "name_chunks": title_texts,
                "multiplier": cluster.multiplier,
                "description": description,
                "description_embeddings": description_embeddings,
                "description_chunks": description_texts,
                "keywords": keyword_res,
                "keywords_reg": keyword_re,
                "mean": np.mean(description_embeddings, axis=0),
            }
        )

    skill_framework_cache.skills.set(cluster_list, Config.skill_framework)

    create_skill_relations()


def create_skill_relations():
    skills = skill_framework_cache.skills.get(Config.skill_framework)
    result: Dict[int, List[Tuple[int, str, float]]] = dict()

    if skills is None:
        raise RuntimeError("You must pre-cache clusters")

    progress = ProgressBar()
    progress.start_sequence(len(skills), "Step 1/1: Linking skills")

    for skill_from in skills:

        result[skill_from["id"]] = []

        embeddings = calculate_complex_similarity(
            skills,
            skill_from["description_embeddings"],
            skill_from["description_chunks"],
            skill_from["description"],
            0.5,
        )

        for skill_with in embeddings:
            if skill_from["id"] == skill_with["id"]:
                continue

            result[skill_from["id"]].append((skill_with["id"], skill_with["name"], skill_with["max"]))

            # print(skill_from["name"] + " -- " + skill_with["name"] + " -- " + str(skill_with["max"]))

        progress.step("Linking skills")

    skill_framework_cache.skill_relations.store(Config.skill_framework, result)


# create_skill_relations()

# import asyncio

# asyncio.run(pre_load_clusters())
