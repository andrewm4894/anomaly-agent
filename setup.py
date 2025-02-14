from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def read_requirements(filename):
    """Read requirements from file, handling comments and empty lines."""
    with open(filename) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("-r")
        ]


# Try to read from requirements.txt first, fall back to requirements.compile
try:
    requirements = read_requirements("requirements.txt")
except FileNotFoundError:
    requirements = read_requirements("requirements.compile")

setup(
    name="anomaly-agent",
    version="0.5.0",
    author="Andrew Maguire",
    author_email="andrewm4894@gmail.com",
    description="A package for detecting anomalies in time series data using LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andrewm4894/anomaly-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
