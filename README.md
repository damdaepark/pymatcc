# pymatcc
Python Materials Conductivity Classifier (Pymatcc) is a classification framework developed to evaluate the ionic conductivity potential of solid-state electrolytes based on their lattice structures. The model currently allows the materials containing Na (sodium) ions given in .cif format.

## How it works
For a given input material, Pymatcc first evaluates its structural descriptors (e.g., orbital field matrix, valence orbital, ...) and encodes them into a feature vector. Based on this feature vector representation, the material is classified into one of the groups made up of structurally similar compounds. Pymatcc then investigates available ionic conductivity records of the compounds within the group, and the potential of input material is assessed. The classifier was trained by unsupervised clustering of Na-ion solide-state electrolytes collected from open databases (e.g., Materials Project, ICSD, etc.). The details about the framework are described in our paper listed in the Citation section below. <br/>

## Setup
Download repository as .zip file and unzip or clone it by:

    $ git clone https://github.com/damdaepark/pymatcc.git "/PATH/TO/DOWNLOAD"
    

Then run the following commands to setup the environment for running `pymatcc`:

    $ cd /PATH/TO/DOWNLOAD
    $ conda create -n ENV_NAME python=3.8.16
    $ conda activate ENV_NAME
    $ pip install -r requirements.txt

If you get any error in installing `annoy` library, try installing:

    $ conda install -c conda-forge python-annoy
    $ pip install -r requirements.txt

Functionality has been verified on Windows (x64) and Linux (x64).

## Requirements
`pymatcc` was developed primarily based on the libraries below: <br/>
- hdbscan==0.8.37 <br/>
- pacmap==0.7.0 <br/>
- pymatgen==2023.8.10 <br/>
- dscribe==2.1.1 <br/>
- numpy==1.24.4 <br/>
- pandas==2.0.3 <br/>
- matplotlib==3.7.5 <br/>
- colorcet==3.1.0 <br/>
- matminer==0.9.0 <br/>

The detailed package requirements are listed in requirements.txt file.

## Usage
Download the repository and run `pymatcc` by:

    $ python ./script/pymatcc.py -f "/ABSOLUTE/PATH/TO/FILENAME.cif" -o "RESULT_FOLDER_NAME"

The results will be saved in ./dat/results/RESULT_FOLDER_NAME <br/>

You can try running an out-of-sample data for test:

    $ python ./script/test.py

The training datasets (2.0 GB in total) will be downloaded during its first execution.

## Data
- Materials Project dataset (ver. 2022.10.28) was used for training the classifier.
- The details of out-of-sample data used for the test run can be found in [Deng, Z. et al. Fundamental investigations on the sodium-ion transport properties of mixed polyanion solid-state battery electrolytes. Nat. Commun. 13, 4470 (2022)](https://www.nature.com/articles/s41467-022-32190-7). We used the data retrieved from its [GitHub repository](https://github.com/caneparesearch/NASICON_KMC_paper_data).

## Contributors
This software was primarily written by Dr. Damdae Park and was advised by Dr. Kyeongsu Kim.

## Citation
If you use `pymatcc` in your research, please consider citing the following work:
	
> Damdae Park, Wonsuk Chung, Byoung Koun Min, Ung Lee, Seungho Yu and Kyeongsu Kim.
> Computational screening of sodium solid electrolytes through unsupervised learning.
> npj Computational Materials. 2024. [URL link TBD]

## Acknowledgement
The development of `pymatcc` was financially supported by the following grants from the National Research Foundation of Korea (NRF) of the Republic of Korea:
- National Supercomputing Center with supercomputing resources including technical support (KSC-2024-CRE-0013).
- institutional research program of Korea Institute of Science and Technology (2E32581).

## License
`pymatcc` is released under the MIT license. The terms of the licenses are as follows:

> MIT License
> 
> Copyright (c) 2024 Damdae Park
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software
> and associated documentation files (the "Software"), to deal in the Software without restriction,
> including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
> and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
> subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
> NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
> IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
> WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
> THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.