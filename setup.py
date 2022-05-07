from setuptools import setup

setup(
    name='grexpy',
    version='0.1.0',
    author='Oluwatosin Olayinka',
    author_email='oaolayin@bu.edu',
    packages=['grexpy'],
    scripts=['scripts/grexpy'],
    license='LICENSE',
    description='Package to calculate genetically regulated expression',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy",
        "statsmodels",
        "pandas",
        "scipy",
        "sklearn",
        "pandas_plink"
    ],
)
