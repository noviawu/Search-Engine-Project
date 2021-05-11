from typing import Dict, Union, List, Generator
import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict


def first_unique_n(iterable, n, min_length=1):
    """
    get first `n` unique elements (greater than length l) from an iterable
    :author: Curtis Wilcox
    :param iterable: thing to get unique elements from
    :param n: how many unique elements to grab
    :param min_length: minimum length of element
    :return: "generator" of first `n` unique elements with length greater than l
             (less if iterable has fewer than `n` unique elements)
    """
    seen = set()
    for element in iterable:
        if element in seen or len(element) < min_length:
            continue
        seen.add(element)
        yield element
        if len(seen) == n:
            return


def load_clean_wapo_with_embedding(
    wapo_jl_path: Union[str, os.PathLike]
) -> Generator[Dict, None, None]:
    """
    load wapo docs as a generator
    :param wapo_jl_path:
    :return: yields each document as a dict
    """
    with open(wapo_jl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            yield json.loads(line)


def parse_wapo_topics(xml_file: str) -> Dict[str, List[str]]:
    """
    parse topics2018.xml
    :param xml_file:
    :return: a dict that maps the topic id to its title. narrative and description
    """
    text = open(xml_file, "r").read()
    topic_mapping = defaultdict(list)

    for xml_str in text.strip().split("\n\n"):
        tree = ET.fromstring(xml_str)
        topic_id = ""
        for child in tree:
            if child.text:
                if child.tag == "num":
                    # get topic id
                    topic_id = child.text.split(":")[-1].strip()
                else:
                    # append others to this topic id
                    topic_mapping[topic_id].append(child.text.strip().split("\n")[-1])
    return topic_mapping


if __name__ == "__main__":
    pass
