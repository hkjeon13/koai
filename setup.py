from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

with open("README.md", mode="r", encoding="utf-8") as readme:
    long_description = readme.read()

setup(
    name='koai',
    version="1.0.8.6",
    description='Korean AI Project',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hkjeon13/koai",
    author="Eddie",
    author_email="hkjeo13@gmail.com",
    zip_safe=False,
    license="MIT",

    py_modules=["koai"],

    python_requires=">=3",

    packages=find_packages("."),
    package_data={"": ["*.json"]},
    include_package_data=True,
    install_requires=[
        "transformers",
        "datasets",
        "seqeval",
        "nltk",
        "rouge_score",
        "evaluate",
        "sklearn",
        "scipy"
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    rust_extensions=[RustExtension("koai/utils/rs_utils", binding=Binding.PyO3)],
)
