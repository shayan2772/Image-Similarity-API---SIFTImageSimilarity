"""
Preprocessing script to extract features from product images.
Run this once to prepare your product database for similarity matching.

Usage:
    python preprocess.py
"""
import os
import sys
import numpy as np
import h5py
from glob import glob
from tqdm import tqdm
from model_util import DeepModel


class ProductFeatureExtractor:
    """Extract features from product images and save for fast lookup."""
    
    def __init__(self, image_directory='./product_images', output_dir='./__generated__'):
        self.image_directory = image_directory
        self.output_dir = output_dir
        self.model = None
        
        # Supported image formats
        self.image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    
    def get_all_images(self):
        """Get all image files from the product directory (including subdirectories)."""
        images = []
        for ext in self.image_extensions:
            # Add ** pattern and recursive=True to search subdirectories
            pattern = os.path.join(self.image_directory, '**', ext)
            images.extend(glob(pattern, recursive=True))
        
        # Sort for consistent ordering
        images.sort()
        return images
    
    def process_images(self, batch_size=32):
        """
        Extract features from all product images.
        
        Args:
            batch_size: Number of images to process at once (adjust based on GPU memory)
        """
        print("=" * 60)
        print("Product Feature Extraction")
        print("=" * 60)
        
        # Get all images
        image_paths = self.get_all_images()
        
        if len(image_paths) == 0:
            print(f"\n❌ Error: No images found in '{self.image_directory}'")
            print(f"   Supported formats: {', '.join(self.image_extensions)}")
            print(f"\n   Please add your product images to '{self.image_directory}' and run again.")
            sys.exit(1)
        
        print(f"\n✓ Found {len(image_paths)} product images")
        print(f"✓ Batch size: {batch_size}")
        
        # Load model
        print("\nLoading MobileNet model...")
        self.model = DeepModel()
        
        # Process images in batches
        print(f"\nExtracting features from {len(image_paths)} images...")
        all_features = []
        metadata = []
        failed_images = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            batch_metadata = []
            
            for img_path in batch_paths:
                try:
                    # Preprocess image
                    img_array = DeepModel.preprocess_image(img_path)
                    batch_images.append(img_array)
                    
                    # Store metadata: [image_id, image_path, filename]
                    filename = os.path.basename(img_path)
                    image_id = os.path.splitext(filename)[0]
                    batch_metadata.append([image_id, img_path, filename])
                    
                except Exception as e:
                    failed_images.append((img_path, str(e)))
                    continue
            
            if len(batch_images) > 0:
                # Extract features for this batch
                batch_array = np.array(batch_images)
                features = self.model.extract_feature(batch_array)
                all_features.append(features)
                metadata.extend(batch_metadata)
        
        if len(all_features) == 0:
            print("\n❌ Error: Failed to process any images")
            sys.exit(1)
        
        # Concatenate all features
        all_features = np.vstack(all_features)
        metadata = np.array(metadata)
        
        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save features as HDF5
        feature_file = os.path.join(self.output_dir, 'products_feature.h5')
        with h5py.File(feature_file, 'w') as h:
            h.create_dataset('data', data=all_features)
        
        # Save metadata as CSV
        metadata_file = os.path.join(self.output_dir, 'products_fields.csv')
        np.savetxt(metadata_file, metadata, delimiter='\t', fmt='%s', encoding='utf-8')
        
        # Print summary
        print("\n" + "=" * 60)
        print("Feature Extraction Complete!")
        print("=" * 60)
        print(f"✓ Successfully processed: {len(metadata)} images")
        print(f"✓ Feature dimension: {all_features.shape[1]}")
        print(f"✓ Total size: {all_features.nbytes / (1024*1024):.2f} MB")
        
        if len(failed_images) > 0:
            print(f"\n⚠ Warning: Failed to process {len(failed_images)} images:")
            for path, error in failed_images[:5]:  # Show first 5
                print(f"  - {os.path.basename(path)}: {error}")
            if len(failed_images) > 5:
                print(f"  ... and {len(failed_images) - 5} more")
        
        print(f"\n📁 Output files:")
        print(f"  - {feature_file}")
        print(f"  - {metadata_file}")
        print("\n✅ You can now start the FastAPI server with: python api.py")
        print("=" * 60)


def main():
    """Main entry point."""
    # Check if product images directory exists
    if not os.path.exists('./product_images'):
        print("❌ Error: './product_images' directory not found")
        print("   Please create it and add your product images first.")
        sys.exit(1)
    
    # Run extraction
    extractor = ProductFeatureExtractor()
    extractor.process_images(batch_size=32)


if __name__ == '__main__':
    main()

