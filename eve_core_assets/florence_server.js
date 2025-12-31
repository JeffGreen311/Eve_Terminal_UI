// Florence-2 HTTP Server for Draw with Eve Analysis
// Provides REST API endpoints for image analysis using Florence-2

import express from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3003;

// Inline Florence analysis function
async function analyzeImageWithFlorence2(imagePath, taskInput = "Detailed Caption") {
    try {
        const { default: Replicate } = await import('replicate');
        const replicate = new Replicate();
        
        const imageBuffer = fs.readFileSync(imagePath);
        const base64Image = imageBuffer.toString('base64');
        const ext = path.extname(imagePath).toLowerCase();
        const mimeTypes = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
            '.gif': 'image/gif', '.webp': 'image/webp', '.svg': 'image/svg+xml'
        };
        const mimeType = mimeTypes[ext] || 'image/jpeg';
        const dataUri = `data:${mimeType};base64,${base64Image}`;

        console.log('🔍 Running Florence-2 model...');
        const output = await replicate.run(
            "lucataco/florence-2-large:da53547e17d45b9cfb48174b2f18af8b83ca020fa76db62136bf9c6616762595",
            { input: { image: dataUri, task: taskInput } }
        );

        return output;
    } catch (error) {
        console.error('❌ Florence-2 Error:', error);
        throw error;
    }
}

// Supported image MIME types (explicit list for clearer errors)
const SUPPORTED_IMAGE_MIME = [
    'image/jpeg', 'image/png', 'image/webp', 'image/gif',
    'image/bmp', 'image/tiff', 'image/svg+xml'
];

// Configure multer for image uploads with improved error clarity
const upload = multer({
    dest: 'uploads/',
    limits: {
        fileSize: 10 * 1024 * 1024, // 10MB limit
    },
    fileFilter: (req, file, cb) => {
        if (!file || !file.mimetype) {
            return cb(new Error('No file or missing MIME type received. Use form field name "image".'), false);
        }
        if (SUPPORTED_IMAGE_MIME.includes(file.mimetype)) {
            return cb(null, true);
        }
        const err = new Error(`Invalid file type '${file.mimetype}'. Only image files are allowed. Supported: ${SUPPORTED_IMAGE_MIME.join(', ')}. Use field name 'image'.`);
        return cb(err, false);
    }
});

// Middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        service: 'Florence-2 Image Analysis Server',
        port: PORT,
        timestamp: new Date().toISOString(),
        api_status: process.env.REPLICATE_API_TOKEN && 
                   !process.env.REPLICATE_API_TOKEN.includes('DISABLED') ? 
                   'active' : 'disabled_for_cost_control'
    });
});

// Main analysis endpoint - accepts multipart form data
app.post('/analyze', upload.single('image'), async (req, res) => {
    try {
        console.log('🔍 Florence-2 Analysis Request Received');
        console.log(`📁 File: ${req.file ? req.file.originalname : 'No file'}`);
        console.log(`📝 Task: ${req.body.task || 'Detailed Caption'}`);

        if (!req.file) {
            return res.status(400).json({
                error: 'No image file provided',
                message: 'Please upload an image file'
            });
        }

        const imagePath = req.file.path;
        const taskInput = req.body.task || 'Detailed Caption';

        // Analyze the image
        const result = await analyzeImageWithFlorence2(imagePath, taskInput);

        // Clean up uploaded file
        fs.unlinkSync(imagePath);

        console.log('✅ Analysis completed successfully');
        res.json({
            success: true,
            analysis: result,
            filename: req.file.originalname,
            task: taskInput,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('❌ Analysis error:', error);
        
        // Clean up uploaded file if it exists
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }

        res.status(500).json({
            success: false,
            error: error.message,
            message: 'Image analysis failed'
        });
    }
});

// Analysis endpoint that accepts base64 image data
app.post('/analyze-base64', async (req, res) => {
    try {
        console.log('🔍 Florence-2 Base64 Analysis Request Received');
        
        const { image_data, task } = req.body;
        
        if (!image_data) {
            return res.status(400).json({
                error: 'No image data provided',
                message: 'Please provide base64 image data'
            });
        }

        // Parse base64 data
        let base64Data = image_data;
        if (image_data.includes(',')) {
            base64Data = image_data.split(',')[1];
        }

        // Create temporary file
        const tempFileName = `temp_${Date.now()}.jpg`;
        const tempPath = path.join('uploads', tempFileName);
        
        // Ensure uploads directory exists
        if (!fs.existsSync('uploads')) {
            fs.mkdirSync('uploads');
        }

        // Write base64 to file
        fs.writeFileSync(tempPath, base64Data, 'base64');

        const taskInput = task || 'Detailed Caption';

        // Analyze the image
        const result = await analyzeImageWithFlorence2(tempPath, taskInput);

        // Clean up temporary file
        fs.unlinkSync(tempPath);

        console.log('✅ Base64 Analysis completed successfully');
        res.json({
            success: true,
            analysis: result,
            task: taskInput,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('❌ Base64 Analysis error:', error);
        
        res.status(500).json({
            success: false,
            error: error.message,
            message: 'Base64 image analysis failed'
        });
    }
});

// FLUX Enhancement endpoint - accepts multipart form data
app.post('/enhance', upload.single('image'), async (req, res) => {
    try {
        console.log('🎨 FLUX Enhancement Request Received');
        console.log(`📁 File: ${req.file ? req.file.originalname : 'No file'}`);
        console.log(`📝 Prompt: ${req.body.prompt || 'enhance and improve this artistic drawing'}`);

        if (!req.file) {
            return res.status(400).json({
                error: 'No image file provided',
                message: 'Please upload an image file'
            });
        }

        const imagePath = req.file.path;
        const prompt = req.body.prompt || 'enhance and improve this artistic drawing';

        // Enhance the image with FLUX
        const result = await enhanceImageWithFLUX(imagePath, prompt);

        // Clean up uploaded file
        fs.unlinkSync(imagePath);

        console.log('✅ Enhancement completed successfully');
        res.json({
            success: true,
            enhanced_image_url: result.enhanced_image_url,
            model_used: result.model_used,
            auto_download: result.auto_download,
            filename: req.file.originalname,
            prompt: prompt,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('❌ Enhancement error:', error);
        
        // Clean up uploaded file if it exists
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }

        res.status(500).json({
            success: false,
            error: error.message,
            message: 'Image enhancement failed'
        });
    }
});

// Status endpoint
app.get('/status', (req, res) => {
    res.json({
        service: 'Florence-2 Image Analysis Server',
        status: 'running',
        port: PORT,
        endpoints: {
            health: 'GET /health',
            analyze: 'POST /analyze (multipart/form-data)',
            analyze_base64: 'POST /analyze-base64 (JSON)',
            status: 'GET /status'
        },
        supported_formats: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff', 'svg'],
        max_file_size: '10MB',
        api_status: process.env.FLORENCE_ANALYSIS_DISABLED === 'true' ? 
                   'disabled_by_user' : 
                   (process.env.REPLICATE_API_TOKEN && !process.env.REPLICATE_API_TOKEN.includes('DISABLED')) ? 
                   'active' : 'token_disabled_for_cost_control',
        timestamp: new Date().toISOString()
    });
});

// Error handling middleware
app.use((error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({
                error: 'File too large',
                message: 'Image file must be smaller than 10MB'
            });
        }
    }
    
    console.error('Server error:', error);
    res.status(500).json({
        error: 'Internal server error',
        message: error.message
    });
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
    console.log('🔍 ========================================');
    console.log('🔍 FLORENCE-2 IMAGE ANALYSIS SERVER');
    console.log('🔍 ========================================');
    console.log(`🔍 Server running on port ${PORT}`);
    console.log('🔍 Local access:');
    console.log(`🔍   http://localhost:${PORT}`);
    console.log(`🔍   http://127.0.0.1:${PORT}`);
    console.log('🔍 Network access:');
    console.log(`🔍   http://0.0.0.0:${PORT}`);
    console.log('🔍 ========================================');
    console.log('🔍 Endpoints:');
    console.log(`🔍   GET  /health - Server health check`);
    console.log(`🔍   GET  /status - Service status`);
    console.log(`🔍   POST /analyze - Image analysis (multipart)`);
    console.log(`🔍   POST /analyze-base64 - Image analysis (base64)`);
    console.log('🔍 ========================================');
    console.log('🔍 Ready for Draw with Eve analysis requests!');
    
    // Check API status on startup
    const apiStatus = process.env.REPLICATE_API_TOKEN && 
                     !process.env.REPLICATE_API_TOKEN.includes('DISABLED') ? 
                     '✅ ACTIVE' : '⚠️ DISABLED (Cost Control)';
    console.log(`🔍 API Status: ${apiStatus}`);
    console.log('🔍 ========================================');
});