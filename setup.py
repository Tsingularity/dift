from setuptools import find_packages, setup

setup(
    name="dift",
    version="0.0.1",
    url="https://diffusionfeatures.github.io/",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    zip_safe=True,
    install_requires=[
        "xformers",
        "torch",
        "accelerate",
        "diffusers",
        "transformers",
    ],
    include_package_data=True,
)
