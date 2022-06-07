import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neurovision", # Replace with your own username
    version="0.0.1",
    author="dq",
    author_email="dq@example.com",
    description="A vision system built with Nengo.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=['neurovision'],
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "nengo",
        "nengo_gui"
    ],
    python_requires=">=3.9",
)