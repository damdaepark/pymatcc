# pymatcc
Python Materials Conductivity Classifier (Pymatcc) is an open-source Python library engineered to rapidly assess the ionic conductivity potential of crystalline compounds based on their lattice structure.

## How it works
For a queried candidate compound, Pymatcc first evaluates its structural descriptors (e.g., orbital field matrix, valence orbital, ...) and encodes them into a feature vector. Based on the feature vector representation, the candidate is classified into a group of structurally similar materials. Here the classifier is a priori trained during the unsupervised clustering of the compounds in a database, from which such a group is also identified. Pymatcc investigates the available ionic conductivity records of the materials within the group and thereby the potential of the candidate is examined. The library currently supports the materials containing Na (sodium) and requires .cif files for input.

## Requirements

## Setup

## Usage


## Data
Materials Project dataset (ver. 2022.10.28) was used for training the prediction model.

## Contributors
This software was primarily written by Dr. Damdae Park and was advised by Dr. Kyeongsu Kim.

## Citation
If you use `pymatcc` in your research, please consider citing the following work:
	
> Damdae Park, Wonsuk Chung, Byoung Koun Min, Ung Lee, Seungho Yu and Kyeongsu Kim.
> Computational screening of sodium solid electrolytes through unsupervised learning.
> npj Computational Materials. 2024. [URL link TBD]

## License
`pymatcc` is released under the MIT license. The terms of the licenses are as follows:

> The MIT License (MIT) Copyright (c) 2011-2012 MIT & LBNL
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
