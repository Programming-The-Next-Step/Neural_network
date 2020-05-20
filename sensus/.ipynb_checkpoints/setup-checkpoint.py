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
      packages=setuptools.find_packages(),
     )
