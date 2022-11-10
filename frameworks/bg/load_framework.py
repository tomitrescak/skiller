from typing import Dict, List
from extract.info import Em


def load_bg_framework() -> List[Em]:

    import csv

    skills: Dict[str, Em] = dict()
    keywords: Dict[str, str] = dict()

    with open("./data/frameworks/bg/SkillCluster.csv") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            skills[row[2]] = Em(int(row[2]), row[0], "", [])

    with open("./data/frameworks/bg/Skill.csv") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            keywords[row[2]] = row[0]

    # add description
    with open("./data/frameworks/bg/SkillClusterDescription.csv") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            if skills[row[3]].description == "":
                skills[row[3]].description = row[2]

    # add keywords
    with open("./data/frameworks/bg/SkillClusters.csv") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            skills[row[0]].keywords.append(keywords[row[1]])

    return list(skills.values())
