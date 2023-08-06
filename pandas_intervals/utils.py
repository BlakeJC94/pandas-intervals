from typing import List

import pandas as pd


def comma_join(x: List[str]) -> str:
    return ", ".join(sorted({i.strip() for n in x if n for i in n.split(",")}))
