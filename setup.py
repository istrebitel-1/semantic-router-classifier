from setuptools import setup


version = "1.0.0"

install_requires = [
    "openai==1.35.7",
    "openpyxl~=3.1.4",
    "pandas==2.2.2",
    "pydantic==2.7.4",
    "pydantic-settings==2.3.4",
    "semantic_router==0.0.48",
    "scikit-learn==1.5.0",
]

setup(
    name="semantic-router-classifier",
    version=version,
    description="Example of classifier based on semantic-router",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    author="Raft",
    author_email="sales@raftds.com",
    include_package_data=True,
    install_requires=install_requires,
    extras_require={
        "code-quality": [
            "black~=24.4.2",
            "flake8~=7.0.0",
            "isort~=5.13.2",
            "mypy~=1.10.0",
            "pylint~=3.2.2",
            "pylint_pydantic~=0.3.0",
            "types-cachetools~=5.3.0.7",
            "types-openpyxl~=3.1.0.20240428",
            "types-setuptools~=68.2.0.2",
            "pandas-stubs~=2.1.4",
        ],
        "testing": ["pytest~=7.4.3", "pytest_asyncio~=0.23.2"],
    },
    packages=[],
    python_requires=">=3.10",
    keywords="semantic-router classifier",
)
