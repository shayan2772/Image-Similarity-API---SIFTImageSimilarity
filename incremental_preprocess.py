"""
Incremental preprocessing module for product images.
Only processes new images that haven't been processed before.
"""
import os
import json
import hashlib
import numpy as np
import h5py
from glob import glob
from datetime import datetime
from typing import List, Dict, Tuple
from model_util import DeepModel


class IncrementalPreprocessor:
    """Handles incremental preprocessing of product images."""
    
    def __init__(self, image_directory='./product_images', output_dir='./__generated__'):
        self.image_directory = image_directory
        self.output_dir = output_dir
        self.model = None
        self.tracking_file = os.path.join(output_dir, 'processed_images.json')
        self.feature_file = os.path.join(output_dir, 'products_feature.h5')
        self.metadata_file = os.path.join(output_dir, 'products_fields.csv')
        
        # Supported image formats
        self.image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    
    def _get_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of a file for change detection."""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return None
    
    def _ensure_directories_exist(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.image_directory,  # product_images folder
            self.output_dir,       # __generated__ folder
            './uploads'            # uploads folder for API file uploads
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"✓ Created directory: {directory}")
    
    def _load_tracking_data(self) -> Dict:
        """Load tracking data of processed images."""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_tracking_data(self, tracking_data: Dict):
        """Save tracking data of processed images."""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
    
    def get_all_images(self) -> List[str]:
        """Get all image files from the product directory (including subdirectories)."""
        images = []
        for ext in self.image_extensions:
            # Add ** pattern and recursive=True to search subdirectories
            pattern = os.path.join(self.image_directory, '**', ext)
            images.extend(glob(pattern, recursive=True))
        images.sort()
        return images
    
    def find_new_images(self) -> Tuple[List[str], List[str]]:
        """
        Find images that haven't been processed or have changed.
        
        Returns:
            Tuple of (new_images, removed_images)
        """
        current_images = self.get_all_images()
        tracking_data = self._load_tracking_data()
        
        # Find new or modified images
        new_or_modified = []
        for img_path in current_images:
            file_hash = self._get_file_hash(img_path)
            if file_hash is None:
                continue
                
            # Check if image is new or modified
            if img_path not in tracking_data or tracking_data[img_path].get('hash') != file_hash:
                new_or_modified.append(img_path)
        
        # Find removed images
        current_paths_set = set(current_images)
        removed = [path for path in tracking_data.keys() if path not in current_paths_set]
        
        return new_or_modified, removed
    
    def _load_existing_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load existing feature data if available."""
        if os.path.exists(self.feature_file) and os.path.exists(self.metadata_file):
            try:
                # Load features
                with h5py.File(self.feature_file, 'r') as h:
                    features = np.array(h['data'])
                
                # Load metadata
                metadata = np.genfromtxt(
                    self.metadata_file,
                    dtype=str,
                    delimiter='\t',
                    encoding='utf-8'
                )
                
                # Ensure metadata is 2D
                if len(metadata.shape) == 1:
                    metadata = metadata.reshape(1, -1)
                
                return features, metadata
            except Exception as e:
                print(f"Warning: Could not load existing data: {e}")
                return None, None
        return None, None
    
    def process_new_images(self, batch_size: int = 32, force_full: bool = False) -> Dict:
        """
        Process only new or modified images.
        
        Args:
            batch_size: Number of images to process at once
            force_full: If True, reprocess all images
            
        Returns:
            Dictionary with processing results
        """
        # Create necessary folders if they don't exist
        self._ensure_directories_exist()
        
        result = {
            'success': False,
            'total_images': 0,
            'new_images': 0,
            'removed_images': 0,
            'failed_images': 0,
            'processing_time': 0,
            'message': ''
        }
        
        start_time = datetime.now()
        
        # Find new and removed images
        if force_full:
            new_images = self.get_all_images()
            removed_images = []
            print("Full reprocessing requested - processing all images")
        else:
            new_images, removed_images = self.find_new_images()
        
        all_current_images = self.get_all_images()
        result['total_images'] = len(all_current_images)
        result['new_images'] = len(new_images)
        result['removed_images'] = len(removed_images)
        
        # Check if there's anything to do
        if len(new_images) == 0 and len(removed_images) == 0:
            result['success'] = True
            result['message'] = 'No new or modified images to process'
            return result
        
        print(f"Found {len(new_images)} new/modified images and {len(removed_images)} removed images")
        
        # Load existing data if not doing full reprocess
        existing_features, existing_metadata = None, None
        if not force_full:
            existing_features, existing_metadata = self._load_existing_data()
        
        # Load model
        if self.model is None:
            print("Loading MobileNet model...")
            self.model = DeepModel()
        
        # Process new images
        new_features = []
        new_metadata = []
        failed_images = []
        
        if len(new_images) > 0:
            print(f"Processing {len(new_images)} new/modified images...")
            
            for i in range(0, len(new_images), batch_size):
                batch_paths = new_images[i:i + batch_size]
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
                    new_features.append(features)
                    new_metadata.extend(batch_metadata)
                    
                    print(f"  Processed batch {i//batch_size + 1}/{(len(new_images)-1)//batch_size + 1}")
        
        # Combine with existing data
        if len(new_features) > 0:
            new_features = np.vstack(new_features)
            new_metadata = np.array(new_metadata)
            
            # Remove old versions of updated images from existing data
            if existing_features is not None and not force_full:
                updated_paths = set([m[1] for m in new_metadata])
                keep_mask = np.array([m[1] not in updated_paths for m in existing_metadata])
                
                if np.any(keep_mask):
                    existing_features = existing_features[keep_mask]
                    existing_metadata = existing_metadata[keep_mask]
                    
                    # Combine with new features
                    all_features = np.vstack([existing_features, new_features])
                    all_metadata = np.vstack([existing_metadata, new_metadata])
                else:
                    all_features = new_features
                    all_metadata = new_metadata
            else:
                all_features = new_features
                all_metadata = new_metadata
        elif existing_features is not None:
            all_features = existing_features
            all_metadata = existing_metadata
        else:
            result['message'] = 'No images to process'
            return result
        
        # Remove deleted images from dataset
        if len(removed_images) > 0 and not force_full:
            removed_set = set(removed_images)
            keep_mask = np.array([m[1] not in removed_set for m in all_metadata])
            all_features = all_features[keep_mask]
            all_metadata = all_metadata[keep_mask]
        
        # Save updated data
        os.makedirs(self.output_dir, exist_ok=True)
        
        with h5py.File(self.feature_file, 'w') as h:
            h.create_dataset('data', data=all_features)
        
        np.savetxt(self.metadata_file, all_metadata, delimiter='\t', fmt='%s', encoding='utf-8')
        
        # Update tracking data
        tracking_data = {}
        for metadata_row in all_metadata:
            img_path = metadata_row[1]
            file_hash = self._get_file_hash(img_path)
            tracking_data[img_path] = {
                'hash': file_hash,
                'processed_at': datetime.now().isoformat(),
                'filename': metadata_row[2]
            }
        self._save_tracking_data(tracking_data)
        
        # Prepare result
        result['success'] = True
        result['failed_images'] = len(failed_images)
        result['processing_time'] = (datetime.now() - start_time).total_seconds()
        result['message'] = f'Successfully processed {len(new_images)} new images'
        
        print(f"\n✓ Incremental update complete!")
        print(f"  Total images in database: {len(all_metadata)}")
        print(f"  New/modified: {len(new_images)}")
        print(f"  Removed: {len(removed_images)}")
        print(f"  Failed: {len(failed_images)}")
        print(f"  Processing time: {result['processing_time']:.2f}s")
        
        return result
    
    def get_status(self) -> Dict:
        """Get current preprocessing status."""
        tracking_data = self._load_tracking_data()
        all_images = self.get_all_images()
        new_images, removed_images = self.find_new_images()
        
        return {
            'total_images': len(all_images),
            'processed_images': len(tracking_data),
            'pending_images': len(new_images),
            'removed_images': len(removed_images),
            'feature_file_exists': os.path.exists(self.feature_file),
            'metadata_file_exists': os.path.exists(self.metadata_file),
            'last_update': max([v.get('processed_at', '') for v in tracking_data.values()], default='Never')
        }

