# SWATgym

SWATgym is a reinforcement learning environment based on the Soil and Water Assessment Tool ([SWAT](https://swat.tamu.edu/)). SWAT is a physics based, continuous time, semi-distributed river basin model that has been widely used to evaluate the effects of crop management decisions on water resources ([Arnold et al., 2012][arnold2012swat]). SWATgym demonstrates the application of reinforcement learning to crop management and enables one to evaluate various decision-making strategies on a full growing season.

Similar to the original SWAT model ([Arnold et al., 1998][arnold1998large]), SWATgym operates on a daily time step and considers various processes including:
- crop growth,
- hydrology, 
- nutrient cycles,
- weather,
- management inputs (fertilizer, irrigation).

## Getting Started
All dependencies are included in the [environment.yml](https://github.com/IBM/SWATgym/blob/main/environment.yml) file.

1. Install SWATgym from source by running
```
git clone https://github.com/IBM/SWATgym
```

2. After cloning, create a virtual environment e.g., using Conda: 
```
conda env create --name swat_env --file=environment.yml
```

3. Activate the environment: `conda activate swat_env`

## References

[Arnold et al., 1998. *Large area hydrologic modeling and assessment part I: model development*. Journal of the American Water Resources Association, 34 (1), 73–89.][arnold1998large]

[Arnold et al., 2012. *SWAT: model use, calibration, and validation*. Transactions of the ASABE, 55 (4), 1491–1508.][arnold2012swat]

## Citing this work

```
@article{madondo2022swatgym,
  author    = {Malvern Madondo and Muneeza Azmat and Kelsey DiPietro and Raya Horesh and Michael Jacobs and Arun Bawa and Raghavan Srinivasan and Fearghal O’Donncha},
  title     = {A SWAT-based Reinforcement Learning Framework for Crop Management},
  year      = {2022},
}
```

[arnold1998large]: https://pubag.nal.usda.gov/download/75/pdf
[arnold2012swat]: https://swat.tamu.edu/media/99051/azdezasp.pdf