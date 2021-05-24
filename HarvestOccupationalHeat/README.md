# Climate risk to rice labour
Code demo to go with [Regional disparities and seasonal differences in climate risk to rice labour](https://doi.org/10.31223/X5SW3N).

The original repo for this example has a DOI [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4746392.svg)](https://doi.org/10.5281/zenodo.4746392).

## Abstract
The 880 million agricultural workers of the world are especially vulnerable to
increasing heat stress due to climate change, affecting the health and income
of individuals, while also decreasing global economic productivity. In this
study, we focus on rice harvests across Asia and estimate the future impact on
labour productivity by considering changes in climate at the time of the annual
harvest. During these specific times of the year, heat stress is often high
compared to the rest of the year. Examining climate simulations of the Coupled
Model Intercomparison Project 6 (CMIP6), we identified that labour productivity
metrics for the rice harvest, based on local wet-bulb globe temperature, are
strongly correlated with global mean near-surface air temperature in the long
term (p<<0.01, R2>0.98 in all models). Limiting global warming to 1.5 °C rather
than 2.0 °C prevents a clear reduction in labour capacity of 1% across all Asia
and 2% across Southeast Asia, affecting the livelihoods of around 100 million
people. Due to differences in mechanization between and within countries, we
find that rice labour is especially vulnerable in Indonesia, the Philippines,
Bangladesh, and the Indian states of West Bengal and Kerala. Our results
highlight the regional disparities and importance in considering seasonal
differences in the estimation of the effect of climate change on labour
productivity and occupational heat-stress.


## Getting started
The notebook has already been evaluated, so [have a look](example.ipynb).

CMIP6 data is retreived from the Pangeo GCS.

A script is provided to setup the conda environment from inside the notebook if required.


## Project Organization
```
├── README.md
├── environment.yml    <- Conda environment specification.
├── example.ipynb <- Example of heat/labour analysis using climate data and crop calendars.
├── example.py    <- Script version of above notebook, used for clean version control.
├── src                <- Source code for use in this project.
│   ├──Labour.py       <- Formulae for assumptions about the effect of WBGT on labour.
│   └──RiceAtlas.py    <- Routine for loading RiceAtlas data.
```
