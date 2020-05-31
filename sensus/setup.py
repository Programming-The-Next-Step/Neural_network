from setuptools import setup

setup(name='sensus',
      version='0.1',
      description='A library for building a simple neural network.',
      url='https://github.com/Programming-The-Next-Step/Neural_network.git',
      author='Eren Asena',
      author_email='eren.asena@student.uva.nl',
      license='MIT',
      packages=['sensus'],
      zip_safe=False,
      classifiers=[
        "Programming Language :: Python :: 3.7.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
      ],
      python_requires='>=3.6',
      install_requires=[
          'numpy'
      ],
      dependency_links=['https://github.com/python/cpython/blob/3.8/Lib/gzip.py',
                       'https://github.com/python/cpython/blob/3.8/Lib/os.py',
                       'https://github.com/python/cpython/blob/3.8/Lib/random.py',
                       'https://github.com/python/cpython/blob/3.8/Lib/urllib/request.py'],
      test_suite='nose.collector',
      tests_require=['nose']
     )
