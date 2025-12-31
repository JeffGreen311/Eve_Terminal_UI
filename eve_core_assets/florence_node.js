// Florence-2 Image Analysis using Node.js Replicate Client
// This script handles image analysis using the official Replicate Node.js library

import fs from 'fs';
import path from 'path';

// Set up environment variable for Replicate API token - DISABLED FOR COST CONTROL
// process.env.REPLICATE_API_TOKEN = 'DISABLED_TO_PREVENT_BILLING';

async function analyzeImageWithFlorence2(imagePath, taskInput = "Detailed Caption") {
    try {
        // Send debug messages to stderr so they don't pollute stdout
        console.error('🔍 Florence-2 Image Analysis Request');
        console.error(`🔍 Image path: ${imagePath}`);
        console.error(`🔍 Task: ${taskInput}`);
        
        // Dynamic import for ESM module
        const { default: Replicate } = await import('replicate');
        
        console.error('🔍 Initializing Replicate client...');
        const replicate = new Replicate();
        
        console.error(`🔍 Reading image from: ${imagePath}`);
        
        // Read the image file as base64
        const imageBuffer = fs.readFileSync(imagePath);
        const base64Image = imageBuffer.toString('base64');
        const mimeType = getMimeType(imagePath);
        const dataUri = `data:${mimeType};base64,${base64Image}`;
        
        console.error(`🔍 Image size: ${imageBuffer.length} bytes`);
        console.error(`🔍 MIME type: ${mimeType}`);
        console.error(`🔍 Task: ${taskInput}`);
        
        const input = {
            image: dataUri,
            task_input: taskInput
        };
        
        console.error('🔍 Running Florence-2 analysis...');
        
        // Run the model using the official client
        const output = await replicate.run(
            "lucataco/florence-2-large:da53547e17d45b9cfb48174b2f18af8b83ca020fa76db62136bf9c6616762595",
            { input }
        );
        
        console.error('🔍 ===== FLORENCE-2 NODE.JS SUCCESS =====');
        console.error('🔍 Analysis complete!');
        
        // Output ONLY the JSON to stdout for Python to parse
        console.log(JSON.stringify(output));
        
        return output;
        
    } catch (error) {
        console.error('❌ Florence-2 Node.js Error:', error);
        
        // Return a structured error response that Python can parse
        const errorResponse = {
            text: {
                '<DETAILED_CAPTION>': `Image analysis failed: ${error.message}. The image file was processed but analysis could not be completed. This may be due to API limitations or network issues.`
            }
        };
        
        // Output ONLY the JSON to stdout for Python to parse
        console.log(JSON.stringify(errorResponse));
        return errorResponse;
    }
}

function getMimeType(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    const mimeTypes = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
        '.svg': 'image/svg+xml',
        '.ico': 'image/x-icon',
        '.heic': 'image/heic',
        '.heif': 'image/heif',
        '.avif': 'image/avif'
    };
    return mimeTypes[ext] || 'image/jpeg';
}

// Export for use in other modules
export { analyzeImageWithFlorence2 };

// If run directly from command line (simplified check)
if (process.argv.length >= 3) {
    const imagePath = process.argv[2];
    const taskInput = process.argv[3] || "Detailed Caption";
    
    if (!imagePath) {
        console.log('Usage: node florence_node.js <image_path> [task_input]');
        process.exit(1);
    }
    
    if (!fs.existsSync(imagePath)) {
        console.error(`❌ Image file not found: ${imagePath}`);
        process.exit(1);
    }
    
    analyzeImageWithFlorence2(imagePath, taskInput)
        .then(result => {
            console.error('✅ Analysis completed successfully');
            process.exit(0);
        })
        .catch(error => {
            console.error('❌ Analysis failed:', error.message);
            process.exit(1);
        });
}