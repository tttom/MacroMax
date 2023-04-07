from setuptools import setup
from pkg_resources import parse_requirements
import m2r2

from macromax import log, __version__

log.info('Reading the package requirements...')
# Require that all packages in requirements.txt are installed prior to this
with open('requirements.txt') as file:
    requirements = [str(req) for req in parse_requirements(file)]

log.info('Converting README.md to rst...')
long_description_rst = m2r2.parse_from_file('README.md')

setup(
    name='macromax',
    version=__version__,
    keywords='light electromagnetic propagation anisotropy magnetic chiral optics Maxwell scattering heterogeneous',
    packages=['macromax'],
    include_package_data=True,
    license='MIT',
    author='Tom Vettenburg',
    author_email='t.vettenburg@dundee.ac.uk',
    description=("Library for solving macroscopic Maxwell's equations for electromagnetic waves in gain-free "
                 + "heterogeneous (bi-)(an)isotropic (non)magnetic materials. This is of particular interest to "
                 + "calculate the light field within complex, scattering, tissues."),
    long_description=long_description_rst,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.7',
    install_requires=requirements,
    extras_require={
        'dev': [
            'build>=0.10.0',
            'twine>=4.0.2',
            'm2r2>=0.3.3',
            'matplotlib>=3.7.0',
            'pprofile>=2.1.0',
            'memory_profiler>=0.61.0',
            'coloredlogs>=15.0.1',
            'sphinx>=6.1.3',
            'sphinx_autodoc_typehints>=1.22',
            'sphinxcontrib_mermaid>=0.8.1',
            'sphinx-rtd-theme>=1.2.0',
            'recommonmark>=0.7.1',
        ],
    },
    zip_safe=True,
    test_suite='nose.collector',
    tests_require=['nose'],
)
