#!/usr/bin/python
# -*- coding: utf-8 -*-

import subprocess

"""
Attempts to automatically make files conform to PEP8 (aggressively).

Runs on files in place.

Best practice: mak these changes as a separate commit from any other changes.
That way you don't confuse "meaningful" code changes vs. PEP8 changes
when someone is looking at history.

Requires autopep8 is installed:

    $ pip install -r requirements-dev.txt
"""


def main():

    files = [
        "dssg/__init__.py",
        "dssg/classifier.py",
        "dssg/machine.py",
        "dssg/platt.py",
        "dssg/util.py",
        "dssg/vectorizer.py",
    ]

    directories = [
        "dssg/model",
        "dssg/webapp",
        "dssg/tests",
    ]

    # autopep8 specific files
    for f in files:
        autopep8_single_file(f)

    for d in directories:
        autopep8_directory_recursively(d)

def autopep8_single_file(path):
    """run autopep8 on a single file"""
    subprocess.call(["autopep8", "--in-place", "--aggressive", path])


def autopep8_directory_recursively(directory):
    """run autopep8 on a directory, recursively"""
    subprocess.call(
        ["autopep8", "--in-place", "--aggressive", directory, "-r"])

if __name__ == "__main__":
    main()




# navigate from `/ushine-learning/scripts` to `/ushine-learning/`
