#!/usr/bin/env python

from distutils.core import setup

#To prepare a new release
#python setup.py sdist upload

setup(name='iceflow',
    version='1.0',
      description='Workflow to derive glacier mass balance from a set of remote-sensing derived DEMs',
    author='GlacierHack',
    author_email='',
    license='MIT',
    url='https://github.com/dshean/iceflow/',
    packages=['iceflow'],
    scripts=['iceflow/glacierhack_pcalign.sh','iceflow/mass_balance.py'])

