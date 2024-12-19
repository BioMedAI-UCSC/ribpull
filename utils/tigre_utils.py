import tigre
import numpy as np
import argparse
import os
import sys
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from tigre.utilities import sample_loader
import tigre.algorithms as algs
import datetime
from PIL import Image

def load_png_projections(projections_dir):
    """Load projections from a directory of PNG files"""
    # List and sort PNG files
    png_files = [f for f in sorted(os.listdir(projections_dir)) 
                 if f.lower().endswith('.png')]
    
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {projections_dir}")
    
    # Load first image to get dimensions
    first_img = Image.open(os.path.join(projections_dir, png_files[0])).convert('L')
    img_shape = np.array(first_img).shape
    
    # Initialize projections array
    projections = np.zeros((len(png_files), img_shape[0], img_shape[1]), dtype=np.float32)
    
    # Load each projection
    for i, file in enumerate(png_files):
        img_path = os.path.join(projections_dir, file)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        projections[i] = np.array(img, dtype=np.float32)
        
    # Normalize to [0, 1] range
    if projections.max() != projections.min():
        projections = (projections - projections.min()) / (projections.max() - projections.min())
    
    return projections

def reconstruct_volume(projections_dir, method='FDK', geometry=None, iterations=50, input_type='npy'):
    """Reconstruct volume from projections using specified method"""
    if input_type == 'npy':
        projections_path = os.path.join(projections_dir, 'projections.npy')
        if not os.path.exists(projections_path):
            raise FileNotFoundError(f"No projections found at {projections_path}")
        projections = np.load(projections_path)
    elif input_type == 'png':
        projections = load_png_projections(projections_dir)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

    angles = np.linspace(0, 2 * np.pi, 100)
    
    if geometry is None:
        geometry = tigre.geometry(mode='cone', default=True)
        # Use the actual dimensions from the loaded projections
        geometry.nDetector = np.array([projections.shape[1], projections.shape[2]])
        geometry.dDetector = [0.8, 0.8]
        geometry.DSD = 1000
        geometry.DSO = 500
    if method.upper() == 'FDK':
        #%% Geometry
        geo = tigre.geometry_default(high_resolution=False)

        #%% Load data and generate projections
        # define angles
        angles = np.linspace(0, 2 * np.pi, 100)
        # Load thorax phantom data
        head = sample_loader.load_head_phantom(geo.nVoxel)
        # generate projections
        projections = tigre.Ax(head, geo, angles)

        reconstruction = algs.fdk(projections, geo, angles)
    elif method.upper() == 'SART':
        reconstruction = tigre.algorithms.sart(
            projections,
            geometry,
            angles,
            niter=iterations
        )
    else:
        raise ValueError(f"Unsupported reconstruction method: {method}")
    
    return reconstruction

def main():
    parser = argparse.ArgumentParser(description='TIGRE DRR generation and reconstruction tool')
    parser.add_argument('operation', choices=['drr', 'reconstruct'], 
                       help='Operation to perform: drr (generate DRRs) or reconstruct')
    parser.add_argument('directory', 
                       help='Directory to save projections (drr) or load projections from (reconstruct)')
    parser.add_argument('--method', choices=['FDK', 'SART'], default='FDK',
                       help='Reconstruction method (default: FDK)')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of iterations for iterative methods (default: 50)')
    parser.add_argument('--volume', 
                       help='Directory containing input DICOM volume for DRR generation')
    parser.add_argument('--output-dicom-dir',
                       help='Directory to save the reconstructed volume as DICOM files')
    parser.add_argument('--input-type', choices=['npy', 'png'], default='npy',
                       help='Type of input projections for reconstruction (default: npy)')
    
    args = parser.parse_args()
    
    os.makedirs(args.directory, exist_ok=True)
    
    if args.operation == 'drr':
        if args.volume is None:
            parser.error("--volume argument is required for drr operation")
        try:
            print("Loading DICOM volume...")
            volume, reference_dicom = load_dicom_volume(args.volume)
            angles = np.linspace(0, 2*np.pi, 360)
            print("Generating DRRs...")
            projections = generate_drr(volume, angles, args.directory)
            print("DRR generation completed successfully")
        except Exception as e:
            print(f"Error generating DRRs: {str(e)}", file=sys.stderr)
            sys.exit(1)
    
    elif args.operation == 'reconstruct':
        try:
            print(f"Starting reconstruction from {args.input_type} projections...")
            reconstruction = reconstruct_volume(
                args.directory,
                method=args.method,
                iterations=args.iterations,
                input_type=args.input_type
            )
            
            if args.output_dicom_dir:
                print("Saving reconstruction as DICOM...")
                save_dicom_volume(reconstruction, args.output_dicom_dir)
                print(f"Reconstruction saved as DICOM files in {args.output_dicom_dir}")
            else:
                output_path = os.path.join(args.directory, f'reconstruction_{args.method}.npy')
                np.save(output_path, reconstruction)
                print(f"Reconstruction saved as numpy array to {output_path}")
                
            print("Reconstruction completed successfully")
        except Exception as e:
            print(f"Error during reconstruction: {str(e)}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()