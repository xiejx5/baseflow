<div align="center">

# baseflow

An open-source Python package for baseflow separation 🔥<br>

</div>
<br>

<div align="center">

Figure Here

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

path = f'{baseflow._path}/example.csv'
Q, date = baseflow.load_streamflow(path)
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
