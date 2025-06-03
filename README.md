# steric-descriptors-alcohol
This is the official repository for the paper "Quantitative Structure–Reactivity Relationship-Guided Mechanistic Study Enables the Optimization of Nitroxyl Radical-Catalyzed Alcohol Oxidation." It provides the code, data, and analysis workflows associated with this publication.
A distinctive feature of this analysis workflow is its streamlined integration of several steps: comprehensive parameter scanning, identification of the most suitable steric descriptors for predicting molecular reactivity using univariate linear regression models, and visualization of model evaluation results. The workflow has been designed so that systematic analyses using steric descriptors can be readily performed. To the best of our knowledge, this study presents the most systematic investigation of %Vbur parameters applied to organic reaction substrates to date.

## Installation
1. Install miniconda from [here](https://docs.anaconda.com/miniconda/)
2. Clone this repository:
```
git clone https://github.com/tanaka-hideya/steric-descriptors-alcohol.git
```
3. Install required packages:
```
cd steric-descriptors-alcohol
conda env create -f environment.yml
```

## Tutorial
To facilitate the reproduction of the analysis workflow used in this study, tutorial notebooks are provided here. 

For further details, please refer to the [tutorial](https://github.com/tanaka-hideya/steric-descriptors-alcohol/tree/main/tutorial).

## Citation
Please cite the following preprint if you use this code or data.

> **"Quantitative Structure–Reactivity Relationship-Guided Mechanistic Study Enables the Optimization of Nitroxyl Radical-Catalyzed Alcohol Oxidation"**  
> Yusuke Sasano, Hideya Tanaka, Shuhei Akutsu, Riki Yamakawa, Shu Saito, Keita Kido, Tsubasa Suzuki, Yuki Tateishi, Shota Nagasawa, Yoshiharu Iwabuchi, Tomoyuki Miyao
> ChemRxiv, 2024
> [DOI: 10.26434/chemrxiv-2025-nlj60](https://chemrxiv.org/engage/chemrxiv/article-details/6821fa04e561f77ed44fc0d3)

## License
This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/tanaka-hideya/steric-descriptors-alcohol/blob/main/LICENSE) for additional details.