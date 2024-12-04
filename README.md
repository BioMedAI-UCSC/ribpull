# Fracture detection in Low-Dose CT images using Neural Skeletons

## Description
The objectives of this research work are as following:
- Perform skeleton extraction from segmenetd rib cages acquired from CT scans in point cloud format using Neural Skeletons
- Perform fracture detection on the skeleton of the rib cage
- Classify fracture between simple classes (non-displaced, displaced, open, closed)
- Investigate fracture detection accuracy and precision under low radiation dose constraints

## Authors
- Manolis Nikolakakis
- Julie Digne
- Razvan Marinescu

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [License](#license)

## Installation
(tba)

```bash
git clone https://github.com/yourusername/ct-shape-analysis.git
cd ct-shape-analysis
```

## Usage
Run test with cave_rib point cloud

```bash
python main.py input_file output_file
```

## Running Tests
Run test with cave_rib point cloud

```bash
python main.py test/cave_rib_input/cave.xyz cave_skeleton.obj
```

## License
(tba)

## Acknowledgments
- Repository based on the following by Mattéo Clémot and Julie Digne: https://github.com/MClemot/SkeletonLearning/tree/main
- We are currently making use of the RibFrac Dataset from the FracNet 2020 challenge 

