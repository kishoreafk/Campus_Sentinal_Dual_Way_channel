"""Setup script for installable package use."""

from setuptools import setup, find_packages

setup(
    name="cctv_interaction_recognition",
    version="1.0.0",
    description="Production-grade CCTV interaction recognition pipeline",
    packages=find_packages(include=["src*", "config*"]),
    python_requires=">=3.10",
    install_requires=[
        # See requirements.txt for the full list — kept here for editable installs.
    ],
)
