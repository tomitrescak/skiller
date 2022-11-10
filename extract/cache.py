from __future__ import annotations
import threading

from typing import Any, Dict, Generic, List, Tuple, TypeVar, cast
from os import makedirs, listdir
from os.path import exists, dirname, isfile, join, basename, splitext
from os import listdir

import pickle
import time

from extract.config import Config
from extract.info import EmbedSource, Skill, SkillCategory, SkillCluster

T = TypeVar("T")


class Map(Generic[T]):
    def __init__(self, root: str, cache: Cache, single: bool = False):
        self.root = root
        self.cache = cache
        self.single = single

        storage: Dict[str, T] = dict()
        self.storage = storage

    def keys(self):
        return self.storage.keys()

    def get_path(self, id: str) -> str:
        if self.single:
            return f"{self.root}.pkl"
        return f"{self.root}/{id}.pkl"

    def get(self, id: str = "item", force: bool = False) -> T | None:
        if not force and not Config.cache:
            return None

        if id in self.storage:
            return self.storage[id]

        path = self.get_path(id)
        cached_role = self.cache.load(path)

        if cached_role is not None:
            self.storage[id] = cached_role

        return cached_role

    def load_all(self) -> List[T] | None:
        values = self.cache.load_all(self.root)

        if values is not None:
            for value in values:
                self.set(cast(T, value["value"]), value["key"])

    def save_all(self):
        for key, value in self.storage.items():
            self.store(key, value)

    def save(self, key: str):
        value = self.get(key)
        if value is not None:
            self.store(key, value)

    def set(self, content: T, id: str = "item", store: bool = True):
        """Stores in memory and pickles the content onto hard drive
        Args:
            id (str): unique id
            content (Any): content
        """

        if not Config.cache:
            return

        self.storage[id] = content
        if store:
            self.store(id, content)

    def store(self, id: str, content: T):
        """Pickles the content onto hard drive
        Args:
            id (str): unique id
            content (Any): content
        """
        self.cache.save(self.get_path(id), content)


# task for a watchdog thread
def watchdog(cache: Cache):
    # run forever
    print(f"Watchdog for cache {cache.root} is running")
    while True:
        # check the version of the cache
        cache.check_cache()
        # block for a moment
        time.sleep(2)


class Cache:
    def __init__(self, root: str, watch: bool = False, store_root: str = ".cache"):
        self.root: str = root
        self.store_root = store_root

        if watch:
            watchdog_task = threading.Thread(target=watchdog, args=[self], name="Watchdog", daemon=True)
            watchdog_task.start()

        self.init_caches()

    def new_version(self):
        return str(int(time.time()))

    def clear_cache(self):
        from shutil import rmtree

        if exists(self.root_path):
            print(f"Removing: " + self.root_path)
            rmtree(self.root_path)
            self.init_caches()

    def init_caches(self):
        pass

    def check_cache(self):
        if not exists(self.root_path):
            self.init_caches()

    @property
    def root_path(self):
        return f"{self.store_root}/{self.root}/"

    def _cached_path(self, path: str):
        return f"./{self.store_root}/{self.root}/{path}"

    def _load_pickle(self, path: str):
        f = open(path, "rb")
        return pickle.load(f)

    def load(self, path: str):
        path = self._cached_path(path)

        if exists(path):
            return self._load_pickle(path)

        return None

    def id_from_file(self, path: str):
        base = basename(path)
        return splitext(base)[0]

    def load_all(self, path: str):
        path = self._cached_path(path)

        if exists(path):
            return [
                {"key": self.id_from_file(f), "value": self._load_pickle(join(path, f))}
                for f in listdir(path)
                if isfile(join(path, f))
            ]
        return None

    def save(self, path: str, content: Any):
        path = self._cached_path(path)

        if not exists(dirname(path)):
            makedirs(dirname(path), exist_ok=True)

        # print("Saving: " + path)
        f = open(path, "wb")
        pickle.dump(content, f)


# class PersistentCache(Cache):
#     def __init__(self):
#         super().__init__("embeddings", False, store_root=".static")

#     def init_caches(self):
#         self.skills = Map[List[EmbedSource]]("skills", self)
#         self.names = Map[List[str]]("names", self)
#         self.descriptions = Map[List[EmbedSource]]("descriptions", self, single=True)
#         self.skill_relations = Map[Dict[int, List[Tuple[int, str, float]]]]("skill_relations", self)

#         self.subject_skills = Map[Dict[str, Subject]]("subject_skills", self)
#         self.skill_info = Map[List[Skill]]("skill_info", self)
#         self.cluster_info = Map[List[SkillCluster]]("cluster_info", self)
#         self.category_info = Map[List[SkillCategory]]("category_info", self)
#         self.job_skills = Map[Dict[int, List[JobSkill]]]("job_skills", self)
#         self.job_related_skills = Map[pd.DataFrame]("job_related_skills", self)


# persistent_cache = PersistentCache()


class SkillFrameworkCache(Cache):
    def __init__(self):
        super().__init__("skills", False, store_root=".static")

    def init_caches(self):
        self.skills = Map[List[EmbedSource]]("skills", self)
        self.names = Map[List[str]]("names", self)
        self.descriptions = Map[List[EmbedSource]]("descriptions", self, single=True)
        self.skill_relations = Map[Dict[int, List[Tuple[int, str, float]]]]("skill_relations", self)
        self.skill_info = Map[List[Skill]]("skill_info", self)
        self.cluster_info = Map[List[SkillCluster]]("cluster_info", self)
        self.category_info = Map[List[SkillCategory]]("category_info", self)

        # self.subject_skills = Map[Dict[str, Subject]]("subject_skills", self)
        # self.job_skills = Map[Dict[int, List[JobSkill]]]("job_skills", self)
        # self.job_related_skills = Map[pd.DataFrame]("job_related_skills", self)


skill_framework_cache = SkillFrameworkCache()
