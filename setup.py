from setuptools import setup, find_packages

setup(
    name="nfn",
    version="3.2.0",
    description="Neural Fractal Network – Architecture de Super-Intelligence Légère par Noyau Fractal Multidimensionnel Condensé",
    author="Philippe-Antoine Robert",
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
    ],
)
