#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name='noregret',
    version='0.0.0.dev17',
    description='No-regret learning dynamics',
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    url='https://github.com/uoftcprg/noregret',
    author=(
        'Universal, Open, Free, and Transparent Computer Poker Research Group'
    ),
    author_email='juhok@cs.cmu.edu',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    keywords=[
        'artificial-intelligence',
        'game',
        'game-theory',
        'imperfect-information-game',
        'online-learning',
        'python',
    ],
    project_urls={
        'Documentation': 'https://noregret.readthedocs.io/en/latest/',
        'Source': 'https://github.com/uoftcprg/noregret',
        'Tracker': 'https://github.com/uoftcprg/noregret/issues',
    },
    packages=find_packages(),
    install_requires=[
        'cupy-cuda13x[ctk]>=14.2.0,<15',
        'gurobipy>=13.0.3,<14',
        'numpy>=2.5.4,<3',
        'open-spiel>=2.0.2,<3',
        'ordered-set>=4.1.0,<5',
        'pokerkit>=0.7.7,<0.8',
        'scipy>=1.18.1,<2',
        'sparse_dot_mkl>=0.9.10,<0.10',
        'tqdm>=4.70.1,<5',
        'ujson>=6.0.0,<7',
    ],
    python_requires='>=3.12',
    package_data={'noregret': ['**/*.json']},
)
