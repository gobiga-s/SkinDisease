"""
Setup script for SkinX Skin Disease Prediction System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="skinx-skin-disease-prediction",
    version="1.0.0",
    author="SkinX Development Team",
    author_email="contact@skinx.ai",
    description="AI-Powered Skin Disease Prediction System using EfficientNet-B3 and BioBERT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/skinx/skin-disease-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Framework :: Flask",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.13.0",
            "torch>=2.0.1+cu118",
            "torchvision>=0.15.2+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "skinx=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["templates/*.html", "static/css/*.css", "static/js/*.js"],
    },
    zip_safe=False,
    keywords="skin disease, medical ai, deep learning, computer vision, nlp, healthcare",
    project_urls={
        "Bug Reports": "https://github.com/skinx/skin-disease-prediction/issues",
        "Source": "https://github.com/skinx/skin-disease-prediction",
        "Documentation": "https://skinx.readthedocs.io/",
    },
)
