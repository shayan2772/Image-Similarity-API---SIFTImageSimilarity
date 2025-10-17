"""
FastAPI Image Similarity Service

This service provides fast image similarity matching for product images.
Pre-computed features are loaded into memory for instant similarity search.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import h5py
from io import BytesIO
import os
import time
import uvicorn

from model_util import DeepModel
from incremental_preprocess import IncrementalPreprocessor


# ============================================================================
# Response Models
# ============================================================================

class PreprocessingResponse(BaseModel):
    """Response for preprocessing endpoint."""
    success: bool = Field(..., description="Whether preprocessing was successful")
    total_images: int = Field(..., description="Total images in product directory")
    new_images: int = Field(..., description="Number of new/modified images processed")
    removed_images: int = Field(..., description="Number of removed images")
    failed_images: int = Field(..., description="Number of images that failed to process")
    processing_time: float = Field(..., description="Processing time in seconds")
    message: str = Field(..., description="Status message")


class PreprocessingStatus(BaseModel):
    """Status of preprocessing system."""
    total_images: int = Field(..., description="Total images in product directory")
    processed_images: int = Field(..., description="Number of processed images")
    pending_images: int = Field(..., description="Number of images pending processing")
    removed_images: int = Field(..., description="Number of removed images")
    feature_file_exists: bool = Field(..., description="Whether feature file exists")
    metadata_file_exists: bool = Field(..., description="Whether metadata file exists")
    last_update: str = Field(..., description="Timestamp of last preprocessing")


class ImageMatch(BaseModel):
    """Single image match result."""
    image_id: str = Field(..., description="Unique identifier for the product image")
    image_path: str = Field(..., description="Full path to the product image")
    filename: str = Field(..., description="Filename of the matched image")
    similarity_score: float = Field(..., description="Similarity score between 0 and 1")


class SimilarityResponse(BaseModel):
    """Response for similarity search."""
    success: bool = Field(..., description="Whether the request was successful")
    matches: List[ImageMatch] = Field(..., description="List of matched images")
    total_compared: int = Field(..., description="Total number of products compared")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    threshold_used: float = Field(..., description="Similarity threshold used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    features_loaded: bool
    total_products: int


class StatsResponse(BaseModel):
    """Statistics response."""
    total_products: int
    feature_dimension: int
    memory_usage_mb: float
    feature_file_size_mb: float


# ============================================================================
# Feature Store
# ============================================================================

class FeatureStore:
    """In-memory feature store for fast similarity search."""
    
    def __init__(self, feature_file: str, metadata_file: str):
        """Load pre-computed features and metadata into memory."""
        print("Loading pre-computed product features...")
        
        if not os.path.exists(feature_file):
            raise FileNotFoundError(
                f"Feature file not found: {feature_file}\n"
                "Please run 'python preprocess.py' first to extract features from product images."
            )
        
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Load features (N x feature_dim array)
        with h5py.File(feature_file, 'r') as h:
            self.features = np.array(h['data'])
        
        # Load metadata (image IDs, paths, filenames)
        self.metadata = np.genfromtxt(
            metadata_file,
            dtype=str,
            delimiter='\t',
            encoding='utf-8',
            skip_header=0
        )
        
        # Ensure metadata is 2D
        if len(self.metadata.shape) == 1:
            self.metadata = self.metadata.reshape(1, -1)
        
        print(f"✓ Loaded {len(self.features)} product features")
        print(f"✓ Feature dimension: {self.features.shape[1]}")
        print(f"✓ Memory usage: {self.features.nbytes / (1024*1024):.2f} MB")
    
    def find_similar(self, query_feature: np.ndarray, threshold: float = 0.84, top_k: int = 10):
        """
        Find similar images using vectorized cosine similarity.
        
        Args:
            query_feature: Feature vector of the query image
            threshold: Minimum similarity threshold (0-1)
            top_k: Maximum number of results to return
            
        Returns:
            List of matching results with similarity scores
        """
        # Compute cosine similarity with all products at once
        similarities = DeepModel.cosine_distance(
            query_feature.reshape(1, -1),
            self.features
        )[0]
        
        # Filter by threshold
        valid_indices = np.where(similarities >= threshold)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Sort by similarity (descending)
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        
        # Return top K matches
        results = []
        for idx in sorted_indices[:top_k]:
            metadata = self.metadata[idx]
            results.append({
                'image_id': str(metadata[0]),
                'image_path': str(metadata[1]),
                'filename': str(metadata[2]) if len(metadata) > 2 else os.path.basename(metadata[1]),
                'similarity_score': float(similarities[idx])
            })
        
        return results


# ============================================================================
# Global State
# ============================================================================

FEATURE_STORE: Optional[FeatureStore] = None
MODEL: Optional[DeepModel] = None
PREPROCESSOR: Optional[IncrementalPreprocessor] = None


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Image Similarity API",
    description="Fast image similarity search for product matching using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model and load features on startup."""
    global MODEL, FEATURE_STORE, PREPROCESSOR
    
    print("\n" + "=" * 60)
    print("Starting Image Similarity API")
    print("=" * 60)
    
    try:
        # Initialize preprocessor
        print("\n1. Initializing preprocessor...")
        PREPROCESSOR = IncrementalPreprocessor()
        
        # Load MobileNet model
        print("\n2. Loading MobileNet model...")
        MODEL = DeepModel()
        
        # Load pre-computed features (if available)
        print("\n3. Loading pre-computed product features...")
        try:
            FEATURE_STORE = FeatureStore(
                feature_file="./__generated__/products_feature.h5",
                metadata_file="./__generated__/products_fields.csv"
            )
            print("✓ Features loaded successfully")
        except FileNotFoundError:
            print("⚠️  No pre-computed features found")
            print("   Use POST /api/v1/preprocess to process product images")
            FEATURE_STORE = None
        
        print("\n" + "=" * 60)
        print("✅ Service Ready!")
        print("=" * 60)
        print(f"📍 API Documentation: http://localhost:8000/docs")
        print(f"📍 Alternative Docs: http://localhost:8000/redoc")
        print(f"📍 Health Check: http://localhost:8000/health")
        print(f"📍 Preprocessing: POST /api/v1/preprocess")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during startup: {e}")
        raise


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page with API information."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Similarity API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #2c3e50; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
            code { background: #34495e; color: #ecf0f1; padding: 2px 6px; border-radius: 3px; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>🖼️ Image Similarity API</h1>
        <p>Fast product image matching using deep learning (MobileNet)</p>
        
        <h2>Quick Links</h2>
        <ul>
            <li><a href="/docs">📚 Interactive API Documentation (Swagger UI)</a></li>
            <li><a href="/redoc">📖 Alternative Documentation (ReDoc)</a></li>
            <li><a href="/health">❤️ Health Check</a></li>
            <li><a href="/stats">📊 Statistics</a></li>
        </ul>
        
        <h2>Main Endpoint</h2>
        <div class="endpoint">
            <strong>POST</strong> <code>/api/v1/find-similar</code><br>
            Upload an image to find similar products
        </div>
        
        <h2>Example Usage (curl)</h2>
        <pre style="background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto;">
curl -X POST "http://localhost:8000/api/v1/find-similar?threshold=0.84&top_k=5" \\
  -F "file=@your_image.jpg"</pre>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify service status.
    
    Returns service health information including model and feature loading status.
    """
    return HealthResponse(
        status="healthy" if MODEL else "degraded",
        model_loaded=MODEL is not None,
        features_loaded=FEATURE_STORE is not None,
        total_products=len(FEATURE_STORE.features) if FEATURE_STORE else 0
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get service statistics.
    
    Returns information about the loaded features, memory usage, and product count.
    """
    if FEATURE_STORE is None:
        raise HTTPException(status_code=503, detail="Service not initialized. Run preprocessing first.")
    
    feature_file = "./__generated__/products_feature.h5"
    file_size_mb = os.path.getsize(feature_file) / (1024 * 1024) if os.path.exists(feature_file) else 0
    
    return StatsResponse(
        total_products=len(FEATURE_STORE.features),
        feature_dimension=FEATURE_STORE.features.shape[1],
        memory_usage_mb=FEATURE_STORE.features.nbytes / (1024 * 1024),
        feature_file_size_mb=file_size_mb
    )


@app.get("/api/v1/preprocess/status", response_model=PreprocessingStatus)
async def get_preprocessing_status():
    """
    Get preprocessing status.
    
    Returns information about processed images, pending images, and last update time.
    Useful for monitoring the preprocessing system.
    """
    if PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Preprocessor not initialized")
    
    status = PREPROCESSOR.get_status()
    return PreprocessingStatus(**status)


@app.post("/api/v1/preprocess", response_model=PreprocessingResponse)
async def run_preprocessing(
    force_full: bool = Query(False, description="Force full reprocessing of all images"),
    batch_size: int = Query(32, ge=1, le=128, description="Batch size for processing")
):
    """
    Run incremental preprocessing of product images.
    
    **This endpoint is designed to be called by a cron job.**
    
    **Behavior:**
    - Only processes new or modified images (unless force_full=true)
    - Automatically detects and removes deleted images
    - Updates the feature database incrementally
    - Reloads features into memory after processing
    
    **Parameters:**
    - **force_full**: If true, reprocess all images (default: false)
    - **batch_size**: Number of images to process at once (default: 32)
    
    **Use Cases:**
    - Cron job: Call this endpoint periodically to keep features up-to-date
    - Manual trigger: Call when you add new product images
    - Full refresh: Use force_full=true to rebuild entire database
    
    **Example cron setup:**
    ```bash
    # Process new images every hour
    0 * * * * curl -X POST http://localhost:8000/api/v1/preprocess
    
    # Full reprocess once per day at 2am
    0 2 * * * curl -X POST "http://localhost:8000/api/v1/preprocess?force_full=true"
    ```
    
    **Returns:**
    - Processing statistics including number of new/removed images
    - Processing time
    - Success status and messages
    """
    if PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Preprocessor not initialized")
    
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Set the model for the preprocessor
        PREPROCESSOR.model = MODEL
        
        # Run preprocessing
        result = PREPROCESSOR.process_new_images(batch_size=batch_size, force_full=force_full)
        
        # Reload features if preprocessing was successful and added new images
        if result['success'] and (result['new_images'] > 0 or result['removed_images'] > 0):
            global FEATURE_STORE
            try:
                print("\n🔄 Reloading features into memory...")
                FEATURE_STORE = FeatureStore(
                    feature_file="./__generated__/products_feature.h5",
                    metadata_file="./__generated__/products_fields.csv"
                )
                print("✓ Features reloaded successfully")
            except Exception as e:
                print(f"⚠️  Warning: Could not reload features: {e}")
        
        return PreprocessingResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Preprocessing failed: {str(e)}"
        )


@app.post("/api/v1/find-similar", response_model=SimilarityResponse)
async def find_similar_images(
    file: UploadFile = File(..., description="Image file to match (JPEG, PNG)"),
    threshold: float = Query(0.84, ge=0.0, le=1.0, description="Similarity threshold (0.84-0.85 recommended)"),
    top_k: int = Query(10, ge=1, le=100, description="Maximum number of matches to return")
):
    """
    Upload an image and find similar products in the database.
    
    **Parameters:**
    - **file**: Image file to upload (JPEG, PNG formats supported)
    - **threshold**: Similarity threshold between 0 and 1 (default: 0.84)
        - 0.84: More matches, less strict
        - 0.845: Balanced (recommended)
        - 0.85+: Fewer matches, more strict
    - **top_k**: Maximum number of similar images to return (default: 10, max: 100)
    
    **Returns:**
    - List of matched products with similarity scores
    - Processing time and statistics
    
    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/find-similar?threshold=0.84&top_k=5" \\
      -F "file=@product_image.jpg"
    ```
    """
    start_time = time.time()
    
    if MODEL is None or FEATURE_STORE is None:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Please ensure preprocessing has been completed."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image file (JPEG, PNG)."
        )
    
    try:
        # Read uploaded file
        contents = await file.read()
        image_bytes = BytesIO(contents)
        
        # Extract features from uploaded image
        query_image = DeepModel.preprocess_image(image_bytes)
        query_feature = MODEL.extract_feature(np.expand_dims(query_image, axis=0))
        
        # Find similar images
        matches = FEATURE_STORE.find_similar(
            query_feature[0],
            threshold=threshold,
            top_k=top_k
        )
        
        # Format response
        result_matches = [ImageMatch(**match) for match in matches]
        
        processing_time = (time.time() - start_time) * 1000
        
        return SimilarityResponse(
            success=True,
            matches=result_matches,
            total_compared=len(FEATURE_STORE.features),
            processing_time_ms=round(processing_time, 2),
            threshold_used=threshold
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

