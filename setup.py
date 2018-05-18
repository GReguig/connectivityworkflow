from setuptools import setup

setup(name="connectivityworkflow",
      version="0.1",
      description="A workflow for graph theory measure calculations from FMRI data",
      url="",
      author="Reguig Ghiles",
      author_email="ghiles.reguig@gmail.com",
      packages=["connectivityworkflow"],
      install_requires=[
              "nipype",
              "nilearn",
              "networkx",
              "pandas",
              "matplotlib",
              "bids",
              "numpy",
              ],
      zip_safe=False
      )
