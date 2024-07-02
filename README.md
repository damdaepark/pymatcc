# pymatcc
Python Materials Conductivity Classifier (Pymatcc) is a classification framework developed to evaluate the ionic conductivity potential of solid-state electrolytes based on their lattice structures. The model currently allows the materials containing Na (sodium) given in .cif format.

## How it works
For a given input material, Pymatcc first evaluates its structural descriptors (e.g., orbital field matrix, valence orbital, ...) and encodes them into a feature vector. Based on this feature vector representation, the material is classified into one of the groups made up of structurally similar compounds. Pymatcc then investigates available ionic conductivity records of the compounds within the group, and the potential of input material is assessed. The classifier was trained by unsupervised clustering of Na-ion solide-state electrolytes collected from open databases (e.g., Materials Project, ICSD, etc.) <br/>

## Requirements
Pymatcc was developed using the libraries specified below. <br/>
hdbscan==0.8.37 <br/>
pacmap==0.7.3 <br/>
pymatgen==2023.8.10 <br/>
dscribe==2.1.1 <br/>
numpy==1.24.4 <br/>
pandas==2.0.3 <br/>
matplotlib==3.7.5 <br/>
colorcet==3.1.0 <br/>
matminer==0.9.0 <br/>

The package requirements are listed in requirements.txt file. Run the following command to install dependencies in your virtual environment:

    $ pip install -r requirements.txt


## Setup
Pymatcc currently does not support pip/conda installation.

## Usage
Download the repository and run pymatcc by

    $ python pymatcc.py -f FILENAME.cif

## Data
Materials Project dataset (ver. 2022.10.28) was used for training the classifier.

## Contributors
This software was primarily written by Dr. Damdae Park and was advised by Dr. Kyeongsu Kim.

## Citation
If you use `pymatcc` in your research, please consider citing the following work:
	
> Damdae Park, Wonsuk Chung, Byoung Koun Min, Ung Lee, Seungho Yu and Kyeongsu Kim.
> Computational screening of sodium solid electrolytes through unsupervised learning.
> npj Computational Materials. 2024. [URL link TBD]

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