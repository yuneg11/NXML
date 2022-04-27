from setuptools import setup, find_packages


with open("nxml/__init__.py") as init_file:
    __version__ = ""
    # extract __version__
    for line in init_file:
        if line.startswith("__version__"):
            exec(line)
            break


with open("README.md") as readme_file:
    readme = readme_file.read()


setup(
    name="nxml",
    version=__version__,
    author="EungGu Yun",
    author_email="yuneg11@gmail.com",
    description="NXML is an eXtension for Machine Learning",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/yuneg11/nxml",
    project_urls={
        "Documentation": "https://yuneg11.github.io/NXML",
        "Source": "https://github.com/yuneg11/nxml",
        "Tracker": "https://github.com/yuneg11/nxml/issues"
    },
    packages=find_packages(include=["nxml", "nxml.*"]),
    package_dir={"nxml": "nxml"},
    # package_data={'': []},
    # include_package_data=True,
    # entry_points={
    #     "console_scripts": [
    #         "nxml=nxml.cli.cli:cli",
    #     ]
    # },
    license="MIT license",
    python_requires=">=3.7",
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
