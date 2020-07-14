![](https://i.imgur.com/nvCDFK0.png)

[![PyPI status](https://img.shields.io/pypi/status/wormpose.svg)](https://pypi.python.org/pypi/wormpose/)
[![PyPI version](https://badge.fury.io/py/wormpose.svg)](https://pypi.python.org/pypi/wormpose/)
[![Build Status](https://travis-ci.org/iteal/wormpose.svg?branch=master)](https://travis-ci.org/iteal/wormpose)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/wormpose.svg)](https://pypi.python.org/pypi/wormpose/)
[![DOI](https://zenodo.org/badge/246706178.svg)](https://zenodo.org/badge/latestdoi/246706178)
[![Maintainability](https://img.shields.io/codeclimate/maintainability/iteal/wormpose)](https://codeclimate.com/github/iteal/wormpose/maintainability)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![](https://i.imgur.com/4b8zz68.gif)

<sub>Results videos adapted from [Open Worm Movement Database](http://movement.openworm.org/) license [CC 4.0](https://creativecommons.org/licenses/by/4.0/legalcode) </sub>


## Overview

The WormPose package estimates the challenging poses of C. elegans (coiled, blurred etc.) in videos where the simple non coiled frames are already labeled. 

We train a convolutional neural network with synthetic worm images so that there is no need for human annotated labels. 

## Get started quickly
[Try the tutorial notebook](https://github.com/iteal/wormpose/blob/master/examples/tutorial_sample_data.ipynb)   <a href="https://colab.research.google.com/github/iteal/wormpose/blob/master/examples/tutorial_sample_data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This notebook goes over the whole WormPose pipeline with some sample data and an already trained model. You can run it in Google Colab.

## Read the documentation
Check the [Documentation website](https://iteal.github.io/wormpose/index.html) for detailed instructions.

## Read the paper
*WormPose: Image synthesis and convolutional networks for pose estimation in C. elegans*  
Laetitia Hebert, Tosif Ahamed, Antonio C. Costa, Liam O’Shaugnessy, Greg J. Stephens  
bioRxiv 2020.07.09.193755; doi: https://doi.org/10.1101/2020.07.09.193755

