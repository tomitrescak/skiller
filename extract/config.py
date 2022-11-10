from typing import Literal
from extract.info import SkillFramework


Strategy = Literal["max"] | Literal["avg"] | Literal["park"]


class Config:
    smoothing: float = 0
    strategy: Strategy = "max"
    subject_skill_threshold: float = 0.5
    cluster_match_threshold: float = 0.45
    job_skill_threshold: float = 0.45  # was 0.6
    cache: bool = True
    max_combinations: int = 100

    skill_framework: SkillFramework = "Asf"
    ignore_clusters = []

    def log(self, text: str):
        ...
        # with open("./log.csv", "a") as file_object:
        #     # Move read cursor to the start of file.
        #     file_object.write(text + "\n")


config = Config()
