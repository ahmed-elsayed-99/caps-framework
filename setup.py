from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="caps-framework",
    version="1.0.0",
    author="Ahmed Elsayed",
    author_email="ahmed_elsayed_mba@outlook.sa",
    description="Competitive-signal Anchored Purchasing-power Segmentation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahmed-elsayed-99/caps-framework",
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ]
    },
    package_data={
        "caps": ["data/raw/*.csv"],
    },
    include_package_data=True,
)