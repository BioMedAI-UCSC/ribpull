# RibPull: Implicit Occupancy Fields and Medial Axis Extraction for CT Ribcage Scans

**Accepted at SPIE Medical Imaging 2026**  
*To be published in the conference proceedings*

## Contributors

- **Emmanouil Nikolakakis** - UC Santa Cruz
- **Amine Ouasfi** - Inria, Univ. Rennes, CNRS, IRISA, M2S
- **Julie Digne** - LIRIS - CNRS - Université Claude Bernard Lyon 1
- **Razvan Marinescu** - UC Santa Cruz

---

## Overview

RibPull is a novel methodology that bridges computational geometry and medical imaging by utilizing implicit occupancy fields to represent CT-scanned ribcages. Our approach enables resolution-independent queries, direct medial axis extraction, and smooth morphological operations that are challenging with traditional discrete voxel-based methods.

## Key Features

- **Neural Occupancy Fields**: Continuous 3D representations that handle sparse and noisy medical imaging data
- **SDF Conversion**: Transforms occupancy fields into signed distance fields for geometric analysis
- **Medial Axis Extraction**: Laplacian-based contraction for robust skeletonization
- **Memory Efficient**: Reduces storage by ~57% (from 4.2 MB to 1.8 MB per scan)
- **Clinical Applications**: Enables fracture detection, scoliosis assessment, and surgical planning

## Method Pipeline

1. **CT Scan Input** → Volumetric computed tomography scan
2. **RibSeg Segmentation** → Binary ribcage segmentation and point cloud extraction
3. **Neural Occupancy Training** → SparseOcc learns implicit surface representation
4. **Mesh Reconstruction** → Isosurface extraction via Marching Cubes
5. **Medial Axis Extraction** → Laplacian-based contraction for skeleton generation

## Dataset

This work uses the **RibSeg dataset**, which extends the RibFrac challenge dataset with:
- 20 manually annotated CT ribcage scans
- High-quality radiologist annotations
- Detailed rib labeling and anatomical centerlines

## Installation
```bash
# Coming soon upon publication
```

## Usage
```python
# Example code will be provided upon release
```

## Citation

If you find this work useful, please cite our paper:
```bibtex
@article{nikolakakis2025ribpull,
  title={RibPull: Implicit Occupancy Fields and Medial Axis Extraction for CT Ribcage Scans},
  author={Nikolakakis, Emmanouil and Ouasfi, Amine and Digne, Julie and Marinescu, Razvan},
  journal={arXiv preprint arXiv:2509.01402},
  year={2025}
}
```

## Acknowledgments

We gratefully acknowledge:
- The authors of **RibSeg** for making their benchmark dataset publicly available
- The creators of the **RibFrac** dataset for their contributions to medical imaging research
- **SparseOcc** methodology by Ouasfi et al. for unsupervised occupancy learning
