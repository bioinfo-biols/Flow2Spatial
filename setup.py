from setuptools import setup

setup(
    name='Flow2Spatial',
    version='0.1',
    author='Ruiqiao He',
    author_email='ruiqiaohe@gmail.com',
    packages=['Flow2Spatial'],
    license="GPL",
    url='http://pypi.python.org/pypi/Flow2Spatial/',
    description='Reconstructing spatial proteomics through transfer learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        "scanpy>=1.9.1",
        "shapely>=1.8.2",
        "cvxpy>=1.1.17",
        "scikit-image",#>=0.19.2
        "anndata",#>=0.8.0
        "scipy",#>=1.7.3
        "numpy",#>=1.22.4
        "pandas",#>=1.3.4
        # "torch>=1.10.0",#+cu102
    ],
    python_requires='>=3.8.10',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'Flow2Spatial=Flow2Spatial.main:f2s_command',
        ]
    }
)
