from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

long_description = long_description.replace("docs/figures/", "https://raw.githubusercontent.com/kyjung2357/specular-differentiation/main/docs/figures/")
long_description = long_description.replace("docs/tutorial.md", "https://github.com/kyjung2357/specular-differentiation/blob/main/docs/tutorial.md")

setup(
    name="specular-differentiation",  
    version="0.11.7",                  
    author="Kiyuob Jung",          
    author_email="kyjung@msu.edu", 
    description="Specular differentiation in normed vector spaces and its applications",
    long_description=long_description, 
    long_description_content_type="text/markdown", 
    url="https://github.com/kyjung2357/specular-differentiation",
    packages=find_packages(),        
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',       
)