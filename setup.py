from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

long_description = long_description.replace(
    "docs/figures/",
    "https://raw.githubusercontent.com/kyjung2357/specular-differentiation/main/docs/figures/"
)

long_description = long_description.replace(
    "docs/README.md",
    "https://github.com/kyjung2357/specular-differentiation/blob/main/docs/README.md"
)

setup(
    long_description=long_description,
    long_description_content_type="text/markdown",
)