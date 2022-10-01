import datetime
import enum
import hashlib
import os
import string
import random
import json

def get_date() -> str:
    """
    Returns the current date in a nice YYYY-MM-DD_H_m_s format
    Returns:
        str
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_dir(*paths) -> str:
    """Creates a dir from a list of directories (like os.path.join), runs os.makedirs and returns the name
    Returns:
        str:
    """
    directory = os.path.join(*paths)
    os.makedirs(directory, exist_ok=True)
    return directory


def hash_text(text: str):
    sha = hashlib.sha1(text)
    sha.hexdigest()


def generate_random_string(length: int = 10):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]