from setuptools import find_packages, setup

setup(
    name="smallbench",
    version="0.1.8",
    packages=find_packages(),
    install_requires=[
        "apropos-ai==0.1.20",
    ],
    author="Josh Purtell",
    author_email="jmvpurtell@gmail.com",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JoshuaPurtell/SmallBench",
    license="MIT",
)
