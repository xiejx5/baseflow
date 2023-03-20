<div align="center">

# baseflow

An open-source Python package for baseflow separation 🔥<br>

</div>
<br>

<div align="center">

![Global Baseflow Index Distribution from 12 Separation Methods](https://user-images.githubusercontent.com/29588684/226364211-3fd46152-3b9a-4de9-8d77-f1b59747a0f4.jpg)

</div>
<br>


## ⚡&nbsp;&nbsp;Usage

### Install
```bash
pip install baseflow
```
<br>


### Example
```python
import baseflow

Q, date = baseflow.load_streamflow(baseflow.example)
b, KGEs = baseflow.separation(Q, date, area=276)
print(f'Best Method: {b.dtype.names[KGEs.argmax()]}')
```
<br>



## Project Structure
The directory structure of baseflow looks like this:
```
├── methods                 <- implements for 12 baseflow separation methods
│
├── recession_analysis      <- tools for estimating recession coefficiency
│
├── param_estimate          <- backward and calibration approaches to estimate other parameters
│
├── comparison              <- an evaluation criterion to comparison different methods
│
├── requirements.txt        <- File for installing baseflow dependencies
│
└── README.md
```
<br>

## 📌&nbsp;&nbsp;Todo


### Nolinear reservoir assumption
- Implement the nolinear reservoir assumption from the [paper](https://github.com/xiejx5/watershed_delineation/releases)
- Employ a time-varing recession coefficiency for baseflow separation
<br>

### Applicable to other time scales
1. The current version only applies to the daily scale
2. The package needs to be updated to support hourly baseflow separation
<br>

## 🚀&nbsp;&nbsp;Publications

### The following articles detail the 12 baseflow separation methods and their evaluation criterion.
- Xie, J., Liu, X., Wang, K., Yang, T., Liang, K., & Liu, C. (2020). Evaluation of typical methods for baseflow separation in the contiguous United States. Journal of Hydrology, 583, 124628. https://doi.org/10.1016/j.jhydrol.2020.124628
