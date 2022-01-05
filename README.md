# Multimedia Data Security

## Introduction

This repo contains material about the following course:

**Name**: Multimedia Data Security

**Teacher**: prof. Giulia Boato, dott. Andrea Montibeller, dott. Federica Lago

**Course**: Computer Science (Master)

**University**: Trento

## Content

You can find the following material in the repo:

```plaintext
├─ 01-lab/ - code and Jupyter notebooks used during the labs.
│ ├── _Images/ - common images for the lab
│ ├── labN-*/ - N labs, each with a subfolder and relevant Jupyter notebooks.
│ ├── ...
├─ 02-challenge/ - mid-term challenge, focused on watermarking processing.
│ ├── competition-day/ - data used during the actual competition day.
│ ├── drawable/ - images used in the final report
│ ├── etc/
│ ├── images/ - a subset of images used during the competition.
│ ├── nonsonoprompt/ - code submitted to the competition.
│ ├── report/ - TeX report of the competition
│ ├── test-results/ - csv results of the tests
│ ├── attacker.py
│ ├── csf.csv
│ ├── main.py
│ ├── nonsonoprompt.npy
│ ├── plot.py
│ └── tester.py
├─ 03-project/ - final project, focused on testing a deep network for image identification
│ ├── drawable/ - images used in the final report
│ ├── noise-extract/ - code dedicated to the extraction of Noiseprints
│ ├── pkg/ - main code 
│ ├── report/ - TeX report
│ ├── verify-size/ - code dedicated to verifying image sizes
│ ├── __init__.py
│ ├── draw_plots.py
│ ├── ncc-parallel.sh
│ ├── ncc_all.py
├─ etc/
  ├── awgn-lecture/ - rendition of a lecture on AWGN in TeX
  └── tex/ - common files used in all TeX files
```

Each subfolder (`01-lab`, `02-challenge`, `03-project`) has two files: `requirements.txt` and `requirements_conda.txt`. They can be used to set up a virtual environment (or Conda environment) for correctly accessing the resources in that folder.

## Authors

* **Paolo Chistè** - *Initial work* - [Paoulus](https://github.com/Paoulus/)
* **Claudio Facchinetti** - *Initial work* - [Facco98](https://github.com/Facco98)
* **Matteo Franzil** - *Initial work* - [mfranzil](https://github.com/mfranzil)
