from setuptools import setup, find_packages


setup(name='ABMT',
      version='1.0.0',
      description='Implementation of Enhancing Diversity in Teacher-Student Networks via Asymmetric branches for Unsupervised Person Re-identification',
      author='Hao Chen',
      author_email='hao.chen@inria.fr',
      url='https://github.com/chenhao2345/ABMT',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Domain Adaptation',
          'Person Re-identification',
          'Teacher-Student Networks',
      ])
