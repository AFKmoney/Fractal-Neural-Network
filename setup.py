from setuptools import setup, find_packages

setup(
    name="nfn",
    version="5.0.0",
    description="Neural Fractal Network — Architecture LM à géométrie fractale et dynamique de phase",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Philippe-Antoine Robert",
    license="Proprietary",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "websockets>=12.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "aiofiles>=23.2.0",
        "httpx>=0.27.0",
    ],
    extras_require={
        "fast": ["tiktoken>=0.6.0"],
        "desktop": ["pywebview>=4.0"],
        "dev": ["pytest>=7.0", "pytest-asyncio>=0.21"],
    },
    entry_points={
        "console_scripts": [
            "nfn-run=run:main",
            "nfn-train=train_agi:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
