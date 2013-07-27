# https://pypi.python.org/pypi/ushine

from setuptools import setup, find_packages
readme = open('README.txt').read()
setup(name='ushine',
      version='0.1',
      author='Kayla Jacobs, Kwang-Sung Jun, Nathan Leiby, Elena Eneva',
      author_email='nathanleiby@gmail.com',
      license='MIT',
      description='Machine learning toolkit - originally built for Ushahidi\'s crowdmapping platform',
      long_description=readme,
      packages=find_packages())