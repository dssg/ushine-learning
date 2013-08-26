# https://pypi.python.org/pypi/ushine

'''setuptools works by triggering subcommands from higher level commands.
The default commands 'install' and 'develop' trigger the following sequences:

install:
  1. build
  2. build_py
  3. install_lib
  4. install_egg_info
  5. egg_info
  6. install_scripts

develop:
  1. egg_info
  2. build_ext
'''

from setuptools import setup, find_packages

readme = open('README.txt').read()
setup(
    name='ushine',
    version='0.1.0',
    author='Kayla Jacobs, Kwang-Sung Jun, Nathan Leiby, Elena Eneva',
    author_email='nathanleiby@gmail.com',
    license='MIT',
    description='Machine learning toolkit - originally built for Ushahidi\'s crowdmapping platform',
    long_description=readme,
    packages=find_packages(),
    install_requires=[
        # 'flask',
        # 'sqlalchemy',
        # 'scikit-learn',
    ],
    dependency_links=[
    ],
    entry_points={
        # 'console_scripts': [],
    },
    tests_require=[
        'nose',
        'pep8',
        'pyflakes',
    ],
    test_suite='test',
)