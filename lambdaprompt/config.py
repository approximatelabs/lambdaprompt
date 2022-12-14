import os
import re
import sys

from dotenv import load_dotenv

if not any(re.findall(r"pytest|py.test", sys.argv[0])):
    load_dotenv()

CONFIG = {}
