# pymatcc
Python Materials Conductivity Classifier (Pymatcc) is an open-source Python library engineered to rapidly assess the ionic conductivity potential of crystalline compounds based on their lattice structure. The library currently supports the materials containing Na (sodium) and .cif input file.

## How it works
For a queried compound, Pymatcc first evaluates its structural descriptors (e.g., orbital field matrix, valence orbital, ...) and encodes them into a feature vector. Based on this feature vector representation, the compound is classified into one of the material groups consisting of structurally similar materials. Pymatcc then investigates available ionic conductivity records of the materials within the group, and the potential of the compound is assessed. It uses a classifier trained through unsupervised clustering of the compounds from available open databases (e.g., Materials Project, ICSD, etc.), during which the material groups were also identified. <br/>


## Requirements

## Setup

## Usage


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