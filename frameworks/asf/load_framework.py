from typing import List

from extract.info import Skill, Em, SkillCategory, SkillCluster
from extract.cache import skill_framework_cache
from extract.config import Config


def load_asf_framework() -> List[Em]:
    import json

    f = open("./frameworks/asf/nsc.json")
    data = json.load(f)

    skills: List[Skill] = []
    clusters: List[SkillCluster] = []
    categories: List[SkillCategory] = []

    # load skills, clusters and categories

    for i, skill in enumerate(data["skillsHierarchy"]):
        clusterId = int(skill[4])
        categoryId = int(skill[5])

        cluster = next((x for x in clusters if x["id"] == clusterId), None)
        if cluster is None:
            clusters.append({"id": clusterId, "name": skill[1], "description": "", "categories": [categoryId]})

        category = next((x for x in clusters if x["id"] == clusterId), None)
        if category is None:
            categories.append({"id": categoryId, "name": skill[2], "description": ""})

        skills.append(
            {"id": i, "name": skill[0], "description": "", "clusters": [clusterId], "keywords": [], "uid": skill[3]}
        )

    start = len(skills)
    clusterId = len(clusters)
    clusters.append({"id": clusterId, "name": "Technology Tools", "description": "", "categories": []})

    for i, tool in enumerate(data["technologyToolDescriptions"]):
        skills.append(
            {
                "id": start + i,
                "name": tool[0],
                "description": tool[1],
                "clusters": [clusterId],
                "keywords": [],
                "uid": tool[2],
            }
        )

    skill_framework_cache.skill_info.set(skills, Config.skill_framework)
    skill_framework_cache.cluster_info.set(clusters, Config.skill_framework)
    skill_framework_cache.category_info.set(categories, Config.skill_framework)

    return [
        *[Em(i, x[0], x[1] + " " + x[2], []) for i, x in enumerate(data["skillsHierarchy"])],
        *[
            Em(i, x[0], x[1], [], 2)
            for i, x in enumerate(data["technologyToolDescriptions"], len(data["skillsHierarchy"]))
        ],
    ]
