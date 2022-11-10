import asyncio
import getopt
import sys

from typing import Any, cast

from extract.embed import calculate_cluster_similarity
from extract.info import SkillFramework
from extract.init import cache_skill_embeddings


def main(argv: Any):

    framework = "Asf"
    input_file = "test.txt"
    will_init = False
    threshold = 0.5

    def help():
        print("============================================")
        print("Skill extractor 1.0")
        print("============================================")
        print("python -m skiller.main [options]")
        print("")
        print("-f (asf|bg) or --framework=(asf|bg) --init : Initialises skill framework (run only once)")
        print("-in <file.txt> or --input=<file.txt>       : Extract skills from text file")
        print("-t <float> or --threshold=<value>          : Threshold (default 0.5)")

    if len(argv) == 0:
        help()
        return

    try:
        opts, _ = getopt.getopt(argv, "hi:f:in:t:", ["init", "framework=", "input=", "threshold="])
    except getopt.GetoptError:
        help()
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            help()
            sys.exit()
        elif opt in ("-i", "--init"):
            will_init = True
        elif opt in ("-f", "--framework"):
            framework = str(arg)
        elif opt in ("-in", "--input"):
            input_file = arg
        elif opt in ("-t", "--threshold"):
            threshold = float(arg)

    if will_init:
        asyncio.run(cache_skill_embeddings(cast(SkillFramework, framework)))

    if input_file != "":

        from os.path import exists

        if not exists(input_file):
            print("File not found")
            return

        with open(input_file) as f:
            content = f.read()
            result = calculate_cluster_similarity(content, threshold)

            for res in result:
                print(f"{res['max']}: {res['name']}")


if __name__ == "__main__":
    main(sys.argv[1:])
