import os
import sys
import shutil
from tkinter import Tk, filedialog
from PIL import Image
import torch
import open_clip
import logging
import subprocess
import tempfile

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
MISSING_DEPS = []

try:
    import cairosvg
    HAVE_CAIROSVG = True
except ImportError:
    HAVE_CAIROSVG = False
    MISSING_DEPS.append("cairosvg")

try:
    import psd_tools
    HAVE_PSD_TOOLS = True
except ImportError:
    HAVE_PSD_TOOLS = False
    MISSING_DEPS.append("psd-tools")

try:
    from pdf2image import convert_from_path
    HAVE_PDF2IMAGE = True
except ImportError:
    HAVE_PDF2IMAGE = False
    MISSING_DEPS.append("pdf2image")

if MISSING_DEPS:
    print("\nSome optional dependencies are missing. To enable all features, install them with:")
    print(f"pip install {' '.join(MISSING_DEPS)}")
    print("\nContinuing with limited functionality...\n")

# Suppress CLIP warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="open_clip")

# Categories for classification
CATEGORIES = [
    "Floral Pattern", 
    "Geometric Design", 
    "Animal Theme", 
    "Abstract Art",
    "Christmas Decoration", 
    "Christmas Ornaments", 
    "Holiday Decorations",
    "Signs",
    "Monogram",
    "Jewlery",
    "Social Media",
    "Home Decor",
    "Font"
]

def load_clip_model():
    """Loads and returns the CLIP model, tokenizer, and preprocessing function."""
    try:
        logger.info("Loading CLIP model...")
        model_name = "ViT-B-32"
        pretrained = "openai"
        logger.debug(f"Using model: {model_name}, pretrained: {pretrained}")
        
        tokenizer = open_clip.get_tokenizer(model_name)
        logger.debug("Tokenizer loaded successfully")
        
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        logger.debug("Model and transforms created successfully")
        
        model = model.eval()
        if torch.cuda.is_available():
            logger.info("CUDA is available, moving model to GPU")
            model = model.cuda()
        else:
            logger.info("CUDA is not available, using CPU")
            
        # Test the model with a simple forward pass
        logger.debug("Testing model with a sample forward pass...")
        test_text = tokenizer(["test"])
        with torch.no_grad():
            test_features = model.encode_text(test_text)
        logger.debug("Model test successful")
        
        return model, tokenizer, preprocess
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {str(e)}", exc_info=True)
        raise

def classify_image(image_path, model, tokenizer, preprocess):
    """Classifies an image using CLIP."""
    try:
        logger.debug(f"Opening image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        logger.debug(f"Image opened successfully: {image.size}")
        
        logger.debug("Preprocessing image...")
        image_input = preprocess(image).unsqueeze(0)
        logger.debug(f"Image preprocessed: shape={image_input.shape}")
        
        if torch.cuda.is_available():
            image_input = image_input.cuda()
            logger.debug("Moved image tensor to GPU")
        
        logger.debug("Tokenizing categories...")
        text_inputs = tokenizer(CATEGORIES)
        logger.debug(f"Categories tokenized: shape={text_inputs.shape}")
        
        if torch.cuda.is_available():
            text_inputs = text_inputs.cuda()
            logger.debug("Moved text tensor to GPU")

        logger.debug("Computing features...")
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            logger.debug(f"Features computed: image={image_features.shape}, text={text_features.shape}")
            
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        logger.debug(f"Similarity scores: {similarity.cpu().numpy()}")
        
        category_idx = torch.argmax(similarity).item()
        confidence = similarity[0][category_idx].item()
        
        logger.info(f"Classification result for {os.path.basename(image_path)}: "
                   f"{CATEGORIES[category_idx]} (confidence: {confidence:.2%})")
        
        return CATEGORIES[category_idx], confidence
    except Exception as e:
        logger.error(f"Error classifying image {image_path}: {str(e)}", exc_info=True)
        return None, 0.0

def convert_ai_to_png(ai_path):
    """Converts AI file to PNG using Inkscape."""
    try:
        png_path = ai_path.rsplit(".", 1)[0] + "_temp.png"
        
        # Try using Inkscape if available
        inkscape_path = shutil.which('inkscape')
        if inkscape_path:
            logger.debug("Using Inkscape for AI conversion")
            try:
                subprocess.run([inkscape_path, ai_path, '--export-filename', png_path], 
                             check=True, capture_output=True)
                if os.path.exists(png_path):
                    return png_path
            except subprocess.CalledProcessError as e:
                logger.error(f"Inkscape conversion failed: {e.stderr.decode() if e.stderr else str(e)}")
        else:
            logger.warning("Inkscape not found. Please install Inkscape for vector file conversion.")
        
        # Only try pdf2image if we have it
        if HAVE_PDF2IMAGE:
            logger.debug("Falling back to pdf2image for AI conversion")
            pdf_path = ai_path.rsplit(".", 1)[0] + "_temp.pdf"
            
            try:
                # Check if GhostScript is available
                gs_path = shutil.which('gswin64c') or shutil.which('gs')
                if not gs_path:
                    logger.error("GhostScript not found. Please install GhostScript for PDF conversion.")
                    return None
                
                # Convert AI to PDF using GhostScript
                subprocess.run([gs_path, '-dNOPAUSE', '-dBATCH', '-sDEVICE=pdfwrite', 
                           f'-sOutputFile={pdf_path}', ai_path], check=True)
                
                # Convert PDF to PNG
                pages = convert_from_path(pdf_path, 500)  # DPI=500 for good quality
                pages[0].save(png_path, 'PNG')
                
                # Cleanup temporary PDF
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                    
                if os.path.exists(png_path):
                    return png_path
            except Exception as e:
                logger.error(f"PDF conversion failed: {str(e)}")
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
        else:
            logger.warning("pdf2image not installed. Install it for better AI file support.")
        
        logger.warning("No suitable converter found for AI file")
        return None
    except Exception as e:
        logger.error(f"Failed to convert AI file: {str(e)}", exc_info=True)
        return None

def convert_cdr_to_png(cdr_path):
    """Converts CDR file to PNG."""
    try:
        png_path = cdr_path.rsplit(".", 1)[0] + "_temp.png"
        
        # Try Inkscape first
        inkscape_path = shutil.which('inkscape')
        if inkscape_path:
            logger.debug("Using Inkscape for CDR conversion")
            subprocess.run([inkscape_path, cdr_path, '--export-filename', png_path], 
                         check=True, capture_output=True)
            if os.path.exists(png_path):
                return png_path
        
        # Try UniConvertor as fallback
        logger.debug("Trying UniConvertor for CDR conversion")
        uniconv_path = shutil.which("uniconvertor")
        if uniconv_path:
            subprocess.run([uniconv_path, cdr_path, png_path], check=True)
            if os.path.exists(png_path):
                return png_path
        
        logger.warning("No suitable converter found for CDR file")
        return None
    except Exception as e:
        logger.error(f"Failed to convert CDR file: {str(e)}", exc_info=True)
        return None

def convert_dxf_to_png(dxf_path):
    """Converts DXF file to PNG."""
    try:
        png_path = dxf_path.rsplit(".", 1)[0] + "_temp.png"
        
        # Try using Inkscape
        inkscape_path = shutil.which('inkscape')
        if inkscape_path:
            logger.debug("Using Inkscape for DXF conversion")
            subprocess.run([inkscape_path, dxf_path, '--export-filename', png_path], 
                         check=True, capture_output=True)
            if os.path.exists(png_path):
                return png_path
        
        logger.warning("No suitable converter found for DXF file")
        return None
    except Exception as e:
        logger.error(f"Failed to convert DXF file: {str(e)}", exc_info=True)
        return None

def convert_vector_to_png(vector_path):
    """Converts vector files to PNG for processing."""
    png_path = vector_path.rsplit(".", 1)[0] + "_temp.png"
    
    try:
        if vector_path.lower().endswith(".svg"):
            if not HAVE_CAIROSVG:
                logger.warning("SVG conversion not available - cairosvg not installed")
                return None
            logger.info(f"Converting SVG: {vector_path}")
            cairosvg.svg2png(url=vector_path, write_to=png_path)
            logger.debug(f"SVG conversion completed: {png_path}")
        elif vector_path.lower().endswith(".psd"):
            if not HAVE_PSD_TOOLS:
                logger.warning("PSD conversion not available - psd-tools not installed")
                return None
            logger.info(f"Converting PSD: {vector_path}")
            psd = psd_tools.PSDImage.open(vector_path)
            psd.composite().save(png_path)
            logger.debug(f"PSD conversion completed: {png_path}")
        elif vector_path.lower().endswith((".ai", ".pdf")):
            logger.info(f"Converting AI/PDF: {vector_path}")
            png_path = convert_ai_to_png(vector_path)
        elif vector_path.lower().endswith(".cdr"):
            logger.info(f"Converting CDR: {vector_path}")
            png_path = convert_cdr_to_png(vector_path)
        elif vector_path.lower().endswith(".dxf"):
            logger.info(f"Converting DXF: {vector_path}")
            png_path = convert_dxf_to_png(vector_path)
        else:
            logger.warning(f"Unsupported vector format: {vector_path}")
            return None

        if png_path and os.path.exists(png_path):
            logger.debug(f"Conversion successful, temporary file created: {png_path}")
            return png_path
        logger.error(f"Conversion failed: temporary file not created")
        return None
    except Exception as e:
        logger.error(f"Conversion failed for {vector_path}: {str(e)}", exc_info=True)
        if png_path and os.path.exists(png_path):
            try:
                os.remove(png_path)
                logger.debug(f"Cleaned up temporary file: {png_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up temporary file: {cleanup_error}")
        return None

def select_directory():
    """Opens a file dialog for directory selection."""
    try:
        root = Tk()
        root.withdraw()
        directory = filedialog.askdirectory(title="Select a folder containing images")
        if directory:
            logger.info(f"Selected directory: {directory}")
            # List contents of selected directory
            files = os.listdir(directory)
            logger.debug(f"Directory contents: {files}")
        else:
            logger.warning("No directory selected")
        return directory
    except Exception as e:
        logger.error(f"Error in directory selection: {str(e)}", exc_info=True)
        return None

def process_directory(directory, model, tokenizer, preprocess):
    """Processes images in the selected directory and its subdirectories."""
    results = {}
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.svg', '.psd', '.ai', '.cdr', '.dxf')
    
    try:
        logger.info(f"Processing directory: {directory}")
        
        # Walk through directory and all subdirectories
        for root, dirs, files in os.walk(directory):
            logger.debug(f"Scanning directory: {root}")
            logger.debug(f"Found {len(files)} files in current directory")
            
            # Process each file in the current directory
            for filename in files:
                if not any(filename.lower().endswith(fmt) for fmt in supported_formats):
                    logger.debug(f"Skipping unsupported file: {filename}")
                    continue
                    
                file_path = os.path.join(root, filename)
                # Get relative path for display
                rel_path = os.path.relpath(file_path, directory)
                logger.info(f"Processing file: {rel_path}")
                
                if filename.lower().endswith(('.svg', '.psd', '.ai', '.cdr', '.dxf')):
                    logger.debug(f"Converting vector file: {filename}")
                    converted_path = convert_vector_to_png(file_path)
                    if converted_path:
                        logger.debug(f"Vector conversion successful, classifying: {converted_path}")
                        category, confidence = classify_image(converted_path, model, tokenizer, preprocess)
                        os.remove(converted_path)
                        logger.debug(f"Temporary file removed: {converted_path}")
                    else:
                        logger.warning(f"Vector conversion failed for: {filename}")
                        category, confidence = "Conversion Failed", 0.0
                else:
                    logger.debug(f"Classifying raster image: {filename}")
                    category, confidence = classify_image(file_path, model, tokenizer, preprocess)
                
                if category:
                    results[rel_path] = (category, confidence)
                    logger.info(f"Successfully classified {rel_path}: {category} ({confidence:.2%})")
                else:
                    results[rel_path] = ("Processing Failed", 0.0)
                    logger.warning(f"Failed to classify {rel_path}")
                    
        logger.info(f"Processed {len(results)} files total")
        return results
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}", exc_info=True)
        return results

def main():
    """Main function guiding the user through directory selection and processing."""
    try:
        print("Starting image classification process...")
        
        # Load CLIP model
        model, tokenizer, preprocess = load_clip_model()
        
        # Select directory
        print("\nPlease select a directory containing images...")
        directory = select_directory()
        if not directory:
            logger.error("No directory selected. Exiting.")
            print("No directory was selected. Exiting.")
            return
        
        # Process images
        print("\nProcessing images (including subdirectories)...")
        results = process_directory(directory, model, tokenizer, preprocess)
        
        # Display results
        if results:
            print("\nCategorization Results:")
            print("-" * 80)
            # Sort results by path for better readability
            for rel_path, (category, confidence) in sorted(results.items()):
                print(f"{rel_path[:40]:<40} -> {category:<25} (Confidence: {confidence:.2%})")
            print("-" * 80)
            print(f"\nProcessed {len(results)} files total")
        else:
            print("\nNo results were generated. Please check the logs for details.")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        print("\nThe program encountered an error. Please check the following:")
        print("1. Make sure all required packages are installed")
        print("2. Ensure you have a valid directory with supported image files")
        print("3. Check if you have enough disk space and memory")
        print(f"\nError details: {str(e)}")

if __name__ == "__main__":
    main()
