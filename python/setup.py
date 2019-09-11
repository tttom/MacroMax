import setuptools
from setuptools import setup

import os
import io


def update_file(file_path, contents):
    if os.path.islink(file_path):
        os.remove(file_path)
    with io.open(file_path, 'w') as output_file:
        output_file.write(contents)


try:
    import pypandoc

    print('pypandoc installed, converting README.md to rst...')
    pypandoc.convert('README.md', 'rst', format='markdown_github', outputfile='README.rst')

    print('Converting README.md also to html...')
    pypandoc.convert('README.md', 'html', format='markdown_github', outputfile='README.html')
    contents = pypandoc.convert('README.md', 'pdf', format='markdown_github', outputfile='README.pdf')
    del contents
except io.ImportError:
    print('Could not import pypandoc, will not regenerate README.')
except io.IOError:
    print('Could not convert README.md, trying to read README.rst directly from disk.')

try:
    long_description_rst = io.open('README.rst', encoding='utf-8').read()
    print('Read README.rst.')
except io.IOError:
    raise io.IOError('Failed to read README.rst file.')

setup(
    name='macromax',
    version='0.0.9',
    keywords='light electromagnetic propagation anisotropy magnetic chiral optics Maxwell scattering heterogeneous',
    packages=['macromax'],
    include_package_data=True,
    license='MIT',
    author='Tom Vettenburg',
    author_email='t.vettenburg@dundee.ac.uk',
    description=('Library for solving macroscopic Maxwell\'s equations for electromagnetic waves in gain-free heterogeneous (bi-)(an)isotropic (non)magnetic materials. This is of particular interest to calculate the light field within complex, scattering, tissues.'),
    long_description=long_description_rst,
    #long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3',
    install_requires=['numpy', 'scipy'],
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose'],
)
