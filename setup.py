from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

GITHUB_URL = "https://github.com/kyjung2357/specular-differentiation"
RAW_URL = "https://raw.githubusercontent.com/kyjung2357/specular-differentiation/main"

long_description = long_description.replace("./docs/figures/", f"{RAW_URL}/docs/figures/")
long_description = long_description.replace("docs/figures/", f"{RAW_URL}/docs/figures/")

long_description = long_description.replace("./docs/", f"{GITHUB_URL}/blob/main/docs/")
long_description = long_description.replace("docs/", f"{GITHUB_URL}/blob/main/docs/")

setup(
    long_description=long_description,
    long_description_content_type="text/markdown",
)