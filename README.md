<div align="center">

# Rapid Watershed Delineation<br>using an Automatic Outlet Relocation Algorithm

Delineating a large number of watersheds for hydrological simulations in the big data era 🔥<br>

</div>
<br>

<div align="center">

Figure Here

</div>
<br>


## ⚡&nbsp;&nbsp;Usage


### Start
- First, download watershed.zip from the [release page](https://github.com/xiejx5/watershed_delineation/releases)
- Next, unzip and open watershed.exe, clip start to execute an example
<br>


## Project Structure
The directory structure of baseflow looks like this:
```
├── bash                    <- Bash scripts
│   ├── setup_conda.sh          <- Setup conda environment
│   └── schedule.sh             <- Schedule execution of many runs
│
├── configs                 <- Hydra configuration files
│   ├── callbacks               <- Callbacks configs
│   ├── datamodule              <- Datamodule configs
│   ├── experiment              <- Experiment configs
│   ├── hparams_search          <- Hyperparameter search configs
│   ├── hydra                   <- Hydra related configs
│   ├── logger                  <- Logger configs
│   ├── model                   <- Model configs
│   ├── trainer                 <- Trainer configs
│   │
│   └── config.yaml             <- Main project configuration file
│
├── methods                 <- implements for 12 baseflow separation methods
│
├── recession_analysis      <- tools for estimating recession coefficiency
│
├── param_estimate          <- backward and calibration approaches to estimate other parameters
│
├── comparison              <- an evaluation criterion to comparison different methods
│
├── requirements.txt        <- File for installing baseflow dependencies
└── README.md
```
<br>