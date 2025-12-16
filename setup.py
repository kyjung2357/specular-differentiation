from setuptools import setup, find_packages
import re

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def replace_relative_links(match):
    link_url = match.group(2) 
    
    if link_url.startswith('#'):
        return match.group(0)

    if link_url.startswith('http') or link_url.startswith('https'):
        return match.group(0)
    
    if link_url.startswith('figures/'):
        github_url = f"https://raw.githubusercontent.com/kyjung2357/specular-differentiation/main/{link_url}"
    else:
        github_url = f"https://github.com/kyjung2357/specular-differentiation/blob/main/{link_url.lstrip('./')}"
        
    return f"[{match.group(1)}]({github_url})"

long_description = re.sub(r'\[(.*?)\]\((.*?)\)', replace_relative_links, long_description)

setup(
    name="specular-differentiation",  
    version="0.0.4",                  
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
    python_requires='>=3.6',       
)