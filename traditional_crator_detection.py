import cv2
import numpy as np
import os

def create_output_dir(output_dir):
    """Create the output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)

def load_image(image_path, output_dir, save_intermediate):
    """Load the image from the specified path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' cannot be loaded.")
    if save_intermediate:
        cv2.imwrite(os.path.join(output_dir, 'output_0_original.png'), image)
    return image

def preprocess_image(image, output_dir, save_intermediate):
    """Convert to grayscale and apply histogram equalization."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if save_intermediate:
        cv2.imwrite(os.path.join(output_dir, 'output_1_grayscale.png'), gray)
    equalized = cv2.equalizeHist(gray)
    if save_intermediate:
        cv2.imwrite(os.path.join(output_dir, 'output_2_equalized.png'), equalized)
    return equalized

def threshold_regions(equalized_image, threshold, inverse=False):
    """Apply binary thresholding to detect regions."""
    if inverse:
        _, binary_mask = cv2.threshold(
            equalized_image,
            threshold,
            255,
            cv2.THRESH_BINARY_INV
        )
    else:
        _, binary_mask = cv2.threshold(
            equalized_image,
            threshold,
            255,
            cv2.THRESH_BINARY
        )
    return binary_mask

def remove_small_objects(binary_mask, kernel_size, min_area, output_dir, prefix, save_intermediate):
    """Remove small objects from the binary mask using morphological operations."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    if save_intermediate:
        cv2.imwrite(os.path.join(output_dir, f'output_{prefix}_opened_mask.png'), opened_mask)
    
    # Find and filter connected components
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(opened_mask)
    components_mask = np.zeros_like(opened_mask)
    regions = []
    
    for label in range(1, num_labels):  # Skip background
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            component_mask = (labels_im == label).astype(np.uint8) * 255
            components_mask = cv2.bitwise_or(components_mask, component_mask)
            centroid = centroids[label]
            regions.append({
                'centroid': tuple(map(int, centroid)),
                'label': label,
                'mask': component_mask
            })
    
    if save_intermediate:
        cv2.imwrite(os.path.join(output_dir, f'output_{prefix}_components_mask.png'), components_mask)
    
    return components_mask, regions

def highlight_regions(original_image, regions_mask, highlight_color, alpha):
    """Highlight regions on the original image with specified color and transparency."""
    color_mask = np.zeros_like(original_image)
    color_mask[regions_mask == 255] = highlight_color
    highlighted_image = cv2.addWeighted(
        src1=original_image,
        alpha=1.0,
        src2=color_mask,
        beta=alpha,
        gamma=0
    )
    return highlighted_image

def save_highlighted_images(original_image, dark_mask, bright_mask, config):
    """Save images with dark and bright regions highlighted."""
    dark_highlighted = highlight_regions(
        original_image,
        dark_mask,
        config['dark_highlight_color'],
        config['highlight_alpha']
    )
    bright_highlighted = highlight_regions(
        original_image,
        bright_mask,
        config['bright_highlight_color'],
        config['highlight_alpha']
    )
    return dark_highlighted, bright_highlighted

def combine_highlighted_images(dark_image, bright_image, output_dir):
    """Combine dark and bright highlighted images side by side."""
    height, width = dark_image.shape[:2]
    combined = np.hstack((dark_image, bright_image))
    cv2.imwrite(os.path.join(output_dir, 'output_combined_highlighted.png'), combined)
    return combined

def match_regions_and_draw_lines(original_image, dark_regions, bright_regions, config):
    """Match bright regions to dark regions and draw connecting lines."""
    line_image = original_image.copy()
    max_distance = config['max_line_length']
    light_dir = config['light_source_direction']
    
    for bright in bright_regions:
        if bright.get('matched'):
            continue
        bright_centroid = bright['centroid']
        potential_matches = []
        
        for dark in dark_regions:
            dark_centroid = dark['centroid']
            distance = np.linalg.norm(np.array(bright_centroid) - np.array(dark_centroid))
            if distance <= max_distance:
                dx = dark_centroid[0] - bright_centroid[0]
                dy = dark_centroid[1] - bright_centroid[1]
                if light_dir == 'top-left' and dx >= 0 and dy >= 0:
                    potential_matches.append((distance, dark))
                elif light_dir == 'top-right' and dx <= 0 and dy >= 0:
                    potential_matches.append((distance, dark))
                elif light_dir == 'bottom-left' and dx >= 0 and dy <= 0:
                    potential_matches.append((distance, dark))
                elif light_dir == 'bottom-right' and dx <= 0 and dy <= 0:
                    potential_matches.append((distance, dark))
                elif light_dir is None:
                    potential_matches.append((distance, dark))
        
        if potential_matches:
            potential_matches.sort(key=lambda x: x[0])
            closest_dark = potential_matches[0][1]
            bright['matched'] = True
            bright['matched_dark_label'] = closest_dark['label']
            cv2.line(
                line_image,
                bright_centroid,
                closest_dark['centroid'],
                config['line_color'],
                thickness=config['line_thickness']
            )
    
    return line_image

def create_full_crater_mask(bright_regions, dark_regions):
    """Create a mask representing full craters by combining matched bright and dark regions."""
    full_crater_mask = np.zeros_like(dark_regions[0]['mask'])
    
    for bright in bright_regions:
        dark_label = bright.get('matched_dark_label')
        if dark_label is not None:
            dark = next((d for d in dark_regions if d['label'] == dark_label), None)
            if dark:
                combined_mask = cv2.bitwise_or(bright['mask'], dark['mask'])
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    all_points = np.vstack(contours)
                    hull = cv2.convexHull(all_points)
                    cv2.drawContours(full_crater_mask, [hull], -1, 255, thickness=-1)
    return full_crater_mask

def overlay_masks(original_image, masks, colors, alphas):
    """Overlay multiple masks on the original image with specified colors and alphas."""
    overlay = original_image.copy()
    for mask, color, alpha in zip(masks, colors, alphas):
        color_mask = np.zeros_like(original_image)
        color_mask[mask == 255] = color
        overlay = cv2.addWeighted(
            src1=overlay,
            alpha=1.0,
            src2=color_mask,
            beta=alpha,
            gamma=0
        )
    return overlay

def generate_video(stages, config, original_image, bright_mask, dark_mask, line_image, full_crater_mask):
    """Generate a video with custom fade cycles based on different stages."""
    video_path = os.path.join(config['output_dir'], config['video_filename'])
    frame_height, frame_width = original_image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_path, fourcc, config['video_fps'], (frame_width, frame_height))
    
    if config['verbose']:
        print(f"Generating video: {video_path}")
    
    # Pre-create images for different stages
    nothing = original_image.copy()
    
    highlight = cv2.addWeighted(
        src1=original_image,
        alpha=1.0,
        src2=bright_mask,
        beta=1.0,
        gamma=0
    )
    highlight = cv2.addWeighted(
        src1=highlight,
        alpha=1.0,
        src2=dark_mask,
        beta=1.0,
        gamma=0
    )
    
    highlight_with_lines = highlight.copy()
    highlight_with_lines = cv2.addWeighted(
        src1=highlight_with_lines,
        alpha=1.0,
        src2=line_image - original_image,
        beta=1.0,
        gamma=0
    )
    
    full_craters_with_highlights_and_lines = cv2.addWeighted(
        src1=original_image,
        alpha=1.0,
        src2=create_full_crater_highlight(original_image, full_crater_mask, config),
        beta=0.25,
        gamma=0
    )
    full_craters_with_highlights_and_lines = cv2.addWeighted(
        src1=full_craters_with_highlights_and_lines,
        alpha=1.0,
        src2=bright_mask,
        beta=1.0,
        gamma=0
    )
    full_craters_with_highlights_and_lines = cv2.addWeighted(
        src1=full_craters_with_highlights_and_lines,
        alpha=1.0,
        src2=dark_mask,
        beta=1.0,
        gamma=0
    )
    full_craters_with_highlights_and_lines = cv2.addWeighted(
        src1=full_craters_with_highlights_and_lines,
        alpha=1.0,
        src2=line_image - original_image,
        beta=1.0,
        gamma=0
    )
    
    full_craters_only = create_full_crater_highlight(original_image, full_crater_mask, config)
    
    stage_images = {
        'nothing': nothing,
        'highlight': highlight,
        'highlight_with_lines': highlight_with_lines,
        'full_craters_with_highlights_and_lines': full_craters_with_highlights_and_lines,
        'full_craters_only': full_craters_only
    }
    
    for cycle in range(config['video_loop_cycles']):
        for stage, duration in config['stage_durations'].items():
            frames_per_stage = int(config['video_fps'] * duration)
            for frame_num in range(frames_per_stage):
                frame = stage_images.get(stage, nothing).copy()
                out_video.write(frame)
                if config['verbose'] and frame_num % 30 == 0:
                    print(f"Cycle {cycle+1}, Stage {stage}, Frame {frame_num+1}/{frames_per_stage}")
    
    out_video.release()
    if config['verbose']:
        print("Video generation complete.")

def create_full_crater_highlight(original_image, full_crater_mask, config):
    """Create an image with full craters highlighted."""
    full_crater_color_mask = np.zeros_like(original_image)
    full_crater_color_mask[full_crater_mask == 255] = config['full_crater_color']
    full_crater_highlighted = cv2.addWeighted(
        src1=original_image,
        alpha=1.0,
        src2=full_crater_color_mask,
        beta=config['full_crater_alpha'],
        gamma=0
    )
    return full_crater_highlighted

def calculate_total_areas(masks, names, verbose=False):
    """Calculate and optionally print the total area of given masks."""
    for mask, name in zip(masks, names):
        area_pixels = cv2.countNonZero(mask)
        if verbose:
            print(f"Total {name} area: {area_pixels} pixels")

def save_intermediate_images(config, images_dict):
    """Save a dictionary of images to the output directory."""
    for filename, image in images_dict.items():
        cv2.imwrite(os.path.join(config['output_dir'], filename), image)

def highlighted_image_with_lines(original_image, bright_mask, dark_mask, line_image):
    """Combine bright and dark masks with lines for highlighting with lines."""
    highlight = cv2.addWeighted(
        src1=original_image,
        alpha=1.0,
        src2=bright_mask,
        beta=1.0,
        gamma=0
    )
    highlight = cv2.addWeighted(
        src1=highlight,
        alpha=1.0,
        src2=dark_mask,
        beta=1.0,
        gamma=0
    )
    highlight_with_lines = cv2.addWeighted(
        src1=highlight,
        alpha=1.0,
        src2=line_image - original_image,
        beta=1.0,
        gamma=0
    )
    return highlight_with_lines

def main():
    # -------------------------------
    # Configuration Parameters
    # -------------------------------
    config = {
        # Path to the input image
        'image_path': 'data/before.png',  # Replace with your image file name

        # Output directory to save results
        'output_dir': 'crator_outputs',

        # Threshold values for binarization
        'dark_threshold': 60,    # Range: 0-255
        'bright_threshold': 200,  # Range: 0-255

        # Kernel size for morphological operations
        'kernel_size': 3,  # Must be a positive odd integer

        # Minimum area (in pixels) to consider a connected component as significant
        'min_area': 20,  # Adjust based on image resolution and region size

        # Colors to highlight regions in BGR format
        'dark_highlight_color': (0, 255, 0),    # Green color for dark regions
        'bright_highlight_color': (0, 0, 255),  # Red color for bright regions
        'line_color': (255, 0, 255),            # Purple color for lines
        'full_crater_color': (0, 255, 255),     # Yellow color for full craters (BGR)

        # Line drawing parameters
        'light_source_direction': None,   # Set to 'top-left', 'top-right', etc., if known
        'line_thickness': 2,              # Thickness of the lines connecting regions
        'max_line_length': 20,            # Maximum length of lines (in pixels)

        # Transparency factors
        'highlight_alpha': 1.0,     # Set to 1.0 for full visibility of highlights
        'line_alpha': 1.0,          # Set to 1.0 for full visibility of lines
        'full_crater_alpha': 0.25,  # Adjusted as per your suggestion

        # Flag to save intermediate images
        'save_intermediate': True,  # Set to False to save only the final output

        # Verbosity level for printing output
        'verbose': True,            # Set to False to suppress print statements

        # Video parameters
        'video_filename': 'regions_fade_cycle.mp4',  # Output video file name
        'video_fps': 30,            # Frames per second
        'video_loop_cycles': 2,     # Number of cycles to include in the video

        # Video stage durations in seconds
        'stage_durations': {
            'nothing': 2,
            'highlight': 2,
            'highlight_with_lines': 2,
            'full_craters_with_highlights_and_lines': 2,
            'full_craters_only': 2,
        },
    }

    # -------------------------------
    # Begin Processing
    # -------------------------------
    try:
        create_output_dir(config['output_dir'])
        original_image = load_image(config['image_path'], config['output_dir'], config['save_intermediate'])
        equalized_image = preprocess_image(original_image, config['output_dir'], config['save_intermediate'])
        
        # Process Dark Regions
        dark_binary = threshold_regions(equalized_image, config['dark_threshold'], inverse=True)
        dark_mask, dark_regions = remove_small_objects(
            dark_binary,
            config['kernel_size'],
            config['min_area'],
            config['output_dir'],
            '3_dark_binary_mask',
            config['save_intermediate']
        )
        
        # Process Bright Regions
        bright_binary = threshold_regions(equalized_image, config['bright_threshold'], inverse=False)
        bright_mask, bright_regions = remove_small_objects(
            bright_binary,
            config['kernel_size'],
            config['min_area'],
            config['output_dir'],
            '6_bright_binary_mask',
            config['save_intermediate']
        )
        
        # Highlight Regions
        dark_highlighted, bright_highlighted = save_highlighted_images(original_image, dark_mask, bright_mask, config)
        if config['save_intermediate']:
            cv2.imwrite(os.path.join(config['output_dir'], 'output_dark_highlighted.png'), dark_highlighted)
            cv2.imwrite(os.path.join(config['output_dir'], 'output_bright_highlighted.png'), bright_highlighted)
        
        combined_highlighted = combine_highlighted_images(dark_highlighted, bright_highlighted, config['output_dir'])
        
        # Match Regions and Draw Lines
        line_image = match_regions_and_draw_lines(original_image, dark_regions, bright_regions, config)
        if config['save_intermediate']:
            cv2.imwrite(os.path.join(config['output_dir'], 'output_lines.png'), line_image)
        
        # Create Full Crater Mask
        full_crater_mask = create_full_crater_mask(bright_regions, dark_regions)
        if config['save_intermediate']:
            cv2.imwrite(os.path.join(config['output_dir'], 'output_full_crater_mask.png'), full_crater_mask)
        
        # Overlay Full Craters
        full_crater_highlighted = create_full_crater_highlight(original_image, full_crater_mask, config)
        cv2.imwrite(os.path.join(config['output_dir'], 'output_full_craters_highlighted.png'), full_crater_highlighted)
        
        # Combine Full Craters with Highlights and Lines
        full_craters_with_highlights_and_lines = cv2.addWeighted(
            src1=full_crater_highlighted,
            alpha=1.0,
            src2=highlighted_image_with_lines(original_image, bright_mask, dark_mask, line_image),
            beta=1.0,
            gamma=0
        )
        cv2.imwrite(
            os.path.join(config['output_dir'], 'output_full_craters_with_highlights_and_lines.png'),
            full_craters_with_highlights_and_lines
        )
        
        # Generate Video
        generate_video(None, config, original_image, bright_mask, dark_mask, line_image, full_crater_mask)
        
        # Calculate Total Areas
        calculate_total_areas(
            [dark_mask, bright_mask, full_crater_mask],
            ['dark region', 'bright region', 'full crater'],
            config['verbose']
        )
        
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    main()
