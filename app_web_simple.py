#!/usr/bin/env python3
"""
Simplified Web Viewer - loads full image instead of tiles for debugging
"""

from flask import Flask, render_template_string, jsonify, send_file, request, redirect, session
from functools import wraps, lru_cache
import numpy as np
import zarr
import json
from pathlib import Path
from scipy.sparse import csr_matrix
from PIL import Image
import io
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# environment variables
APP_PASSWORD = os.environ.get('APP_PASSWORD', 'changeme')
DATA_DIR = Path(os.environ.get('DATA_DIR', './data'))

# paths
TILES_DIR = DATA_DIR / 'AB01'
IMAGE_PATH = DATA_DIR / 'AB01_he_scaled.tiff'

# password protection decorator
def require_password(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # check if already authenticated
        if session.get('authenticated'):
            return f(*args, **kwargs)

        # check password from query param or form
        password = request.args.get('password') or request.form.get('password')
        if password == APP_PASSWORD:
            session['authenticated'] = True
            return f(*args, **kwargs)

        # show login page
        return render_template_string(LOGIN_TEMPLATE)
    return decorated_function

# cached data - load once and reuse (but NOT dense segmentation - too big!)
FULL_IMAGE = None
ZARR_DATA = None
SEGMENTATION = None

def get_full_image():
    global FULL_IMAGE
    if FULL_IMAGE is None:
        if not IMAGE_PATH.exists():
            raise FileNotFoundError(f"Image file not found at {IMAGE_PATH}")
        from tifffile import imread
        print(f"Loading full image from {IMAGE_PATH}...")
        img = imread(IMAGE_PATH)
        # downscale to half for web viewing
        pil_img = Image.fromarray(img)
        new_size = (pil_img.width // 2, pil_img.height // 2)
        pil_img = pil_img.resize(new_size, Image.LANCZOS)
        FULL_IMAGE = np.array(pil_img)
        print(f"Full image loaded and cached: {FULL_IMAGE.shape}")
    return FULL_IMAGE

def load_zarr_expression():
    """load zarr expression data (cached)"""
    global ZARR_DATA
    if ZARR_DATA is None:
        zarr_path = TILES_DIR / 'expression.zarr'
        print(f"Loading zarr data from {zarr_path}...")
        ZARR_DATA = zarr.open(str(zarr_path), mode='r')
        print(f"Zarr data loaded and cached")
    return ZARR_DATA

def load_segmentation():
    """load segmentation mask (cached as sparse matrix)"""
    global SEGMENTATION
    if SEGMENTATION is None:
        seg_path = TILES_DIR / 'segmentation.npz'
        print(f"Loading segmentation from {seg_path}...")
        data = np.load(seg_path)
        SEGMENTATION = csr_matrix(
            (data['data'], data['indices'], data['indptr']),
            shape=tuple(data['shape'])
        )
        print(f"Segmentation loaded and cached: {SEGMENTATION.shape}")
    return SEGMENTATION

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Login - Spatial Viewer</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            background: #2c3e50;
        }
        .login-box {
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            width: 300px;
        }
        h1 {
            font-size: 24px;
            margin-bottom: 20px;
            color: #2c3e50;
            text-align: center;
        }
        input[type="password"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
            margin-bottom: 15px;
        }
        button {
            width: 100%;
            padding: 12px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover { background: #2980b9; }
    </style>
</head>
<body>
    <div class="login-box">
        <h1>Spatial Viewer</h1>
        <form method="POST">
            <input type="password" name="password" placeholder="Enter password" autofocus required>
            <button type="submit">Login</button>
        </form>
    </div>
</body>
</html>
'''

UPLOAD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Upload Data - Spatial Viewer</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            padding: 40px;
            background: #2c3e50;
            color: white;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: #34495e;
            padding: 30px;
            border-radius: 8px;
        }
        h1 { margin-bottom: 20px; }
        .upload-section {
            background: #2c3e50;
            padding: 20px;
            border-radius: 4px;
            margin: 20px 0;
        }
        h2 { font-size: 18px; margin-bottom: 10px; }
        input[type="file"] { margin: 10px 0; }
        button {
            padding: 10px 20px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover { background: #2980b9; }
        .info {
            background: #27ae60;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .warning {
            background: #e74c3c;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }
        pre {
            background: #1a1a1a;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Upload Data Files</h1>
        <div class="info">
            <strong>Instructions:</strong> Upload your data files to the Railway volume.
            After uploading, remove this upload endpoint and redeploy.
        </div>

        <div class="upload-section">
            <h2>1. Upload TIFF Image</h2>
            <form action="/upload/image" method="POST" enctype="multipart/form-data">
                <input type="file" name="file" accept=".tiff,.tif" required>
                <button type="submit">Upload Image</button>
            </form>
        </div>

        <div class="upload-section">
            <h2>2. Upload NPZ File (segmentation.npz)</h2>
            <form action="/upload/npz" method="POST" enctype="multipart/form-data">
                <input type="file" name="file" accept=".npz" required>
                <button type="submit">Upload NPZ</button>
            </form>
        </div>

        <div class="upload-section">
            <h2>3. Upload JSON File (gene_index.json)</h2>
            <form action="/upload/json" method="POST" enctype="multipart/form-data">
                <input type="file" name="file" accept=".json" required>
                <button type="submit">Upload JSON</button>
            </form>
        </div>

        <div class="upload-section">
            <h2>4. Upload Zarr Directory</h2>
            <div class="warning">
                <strong>Note:</strong> Zarr directories need to be uploaded as a zip file.
                Zip your expression.zarr folder first, then upload it here.
            </div>
            <form action="/upload/zarr" method="POST" enctype="multipart/form-data">
                <input type="file" name="file" accept=".zip" required>
                <button type="submit">Upload Zarr (as ZIP)</button>
            </form>
        </div>

        <div class="upload-section">
            <h2>Current Status</h2>
            <pre>{{ status }}</pre>
        </div>
    </div>
</body>
</html>
'''

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Spatial Viewer</title>
    <meta charset="utf-8">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; display: flex; height: 100vh; }
        #sidebar { width: 300px; background: #2c3e50; color: white; padding: 20px; overflow-y: auto; }
        #map { flex: 1; }
        h1 { font-size: 20px; margin-bottom: 10px; }
        label { display: block; margin-top: 15px; font-weight: 600; }
        input, button { width: 100%; padding: 8px; margin-top: 5px; border-radius: 4px; border: none; }
        button { background: #3498db; color: white; cursor: pointer; margin-top: 10px; }
        button:hover { background: #2980b9; }
        #stats { background: #34495e; padding: 15px; border-radius: 4px; margin-top: 20px; font-size: 13px; }
        #loading { position: fixed; top: 20px; right: 20px; background: #3498db; color: white;
                   padding: 10px 20px; border-radius: 4px; display: none; z-index: 10000; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>Spatial Viewer</h1>
        <label>Gene:</label>
        <input type="text" id="gene-search" list="gene-list" placeholder="Type gene name...">
        <datalist id="gene-list"></datalist>
        <button id="load-gene">Load Expression</button>
        <button id="clear-gene">Clear</button>

        <label>Expression Opacity:</label>
        <input type="range" id="opacity" min="0" max="100" value="100">
        <span id="opacity-val">100%</span>

        <label style="margin-top: 20px;">
            <input type="checkbox" id="show-boundaries" style="width: auto; margin-right: 5px;">
            Show Cell Boundaries
        </label>

        <label>Boundary Opacity:</label>
        <input type="range" id="boundary-opacity" min="0" max="100" value="100">
        <span id="boundary-opacity-val">100%</span>

        <div id="stats">Loading...</div>
    </div>
    <div id="map"></div>
    <div id="loading">Loading...</div>

    <script>
        let genes = [];
        let expressionLayer = null;
        let boundaryLayer = null;
        let currentGene = null;

        const map = L.map('map', {
            crs: L.CRS.Simple,
            minZoom: -2,
            maxZoom: 2
        });

        async function init() {
            // load metadata
            const metaResp = await fetch('/api/metadata');
            const metadata = await metaResp.json();

            // load genes
            const geneResp = await fetch('/api/genes');
            genes = await geneResp.json();
            const datalist = document.getElementById('gene-list');
            genes.forEach(g => {
                const opt = document.createElement('option');
                opt.value = g;
                datalist.appendChild(opt);
            });

            document.getElementById('stats').innerHTML =
                `<strong>Ready!</strong><br>${metadata.genes.n_cells.toLocaleString()} cells<br>${metadata.genes.n_genes.toLocaleString()} genes`;

            // load and display image
            const imgResp = await fetch('/api/full_image');
            const imgBlob = await imgResp.blob();
            const imgUrl = URL.createObjectURL(imgBlob);

            const img = new Image();
            img.onload = () => {
                const bounds = [[0, 0], [img.height, img.width]];
                L.imageOverlay(imgUrl, bounds).addTo(map);
                map.fitBounds(bounds);
            };
            img.src = imgUrl;
        }

        async function loadExpression(gene) {
            document.getElementById('loading').style.display = 'block';

            try {
                console.log('Loading expression for:', gene);

                const resp = await fetch(`/api/expression/${gene}`);
                const data = await resp.json();

                if (data.error) {
                    alert(data.error);
                    return;
                }

                const stats = data.stats;
                document.getElementById('stats').innerHTML =
                    `<strong>${gene}</strong><br>` +
                    `Expressing: ${stats.n_expressing.toLocaleString()}/${stats.total_cells.toLocaleString()}<br>` +
                    `Mean: ${stats.mean.toFixed(2)}<br>` +
                    `Max: ${stats.max.toFixed(2)}`;

                // get expression overlay
                console.log('Fetching expression overlay...');
                const overlayResp = await fetch(`/api/expression_overlay/${gene}`);
                console.log('Overlay response status:', overlayResp.status);

                if (!overlayResp.ok) {
                    const errorText = await overlayResp.text();
                    console.error('Overlay error:', errorText);
                    alert('Failed to load expression overlay: ' + errorText);
                    return;
                }

                const overlayBlob = await overlayResp.blob();
                console.log('Overlay blob size:', overlayBlob.size);
                const overlayUrl = URL.createObjectURL(overlayBlob);

                if (expressionLayer) {
                    map.removeLayer(expressionLayer);
                }

                const img = new Image();
                img.onload = () => {
                    console.log('Expression overlay image loaded:', img.width, 'x', img.height);
                    const bounds = [[0, 0], [img.height, img.width]];
                    const opacity = document.getElementById('opacity').value / 100;
                    expressionLayer = L.imageOverlay(overlayUrl, bounds, {
                        opacity: opacity,
                        interactive: false
                    });
                    expressionLayer.addTo(map);
                    console.log('Expression overlay added to map');
                };
                img.onerror = () => {
                    console.error('Failed to load expression overlay image');
                    alert('Failed to display expression overlay image');
                };
                img.src = overlayUrl;

            } catch (error) {
                console.error('Error loading expression:', error);
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        document.getElementById('load-gene').onclick = () => {
            const gene = document.getElementById('gene-search').value.trim();
            if (gene && genes.includes(gene)) {
                currentGene = gene;
                loadExpression(gene);
            } else {
                alert('Please select a valid gene');
            }
        };

        document.getElementById('clear-gene').onclick = () => {
            if (expressionLayer) {
                map.removeLayer(expressionLayer);
                expressionLayer = null;
            }
            currentGene = null;
            document.getElementById('gene-search').value = '';
        };

        document.getElementById('opacity').oninput = (e) => {
            const val = e.target.value;
            document.getElementById('opacity-val').textContent = val + '%';
            if (expressionLayer) {
                expressionLayer.setOpacity(val / 100);
            }
        };

        document.getElementById('boundary-opacity').oninput = (e) => {
            const val = e.target.value;
            document.getElementById('boundary-opacity-val').textContent = val + '%';
            if (boundaryLayer) {
                boundaryLayer.setOpacity(val / 100);
            }
        };

        document.getElementById('show-boundaries').onchange = async (e) => {
            if (e.target.checked) {
                // load boundaries
                document.getElementById('loading').style.display = 'block';
                try {
                    console.log('Loading cell boundaries...');
                    const resp = await fetch('/api/cell_boundaries');
                    const blob = await resp.blob();
                    const url = URL.createObjectURL(blob);

                    const img = new Image();
                    img.onload = () => {
                        const bounds = [[0, 0], [img.height, img.width]];
                        const opacity = document.getElementById('boundary-opacity').value / 100;
                        boundaryLayer = L.imageOverlay(url, bounds, {
                            opacity: opacity,
                            interactive: false
                        });
                        boundaryLayer.addTo(map);
                        console.log('Cell boundaries added');
                    };
                    img.src = url;
                } catch (error) {
                    console.error('Error loading boundaries:', error);
                    alert('Failed to load boundaries: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            } else {
                // remove boundaries
                if (boundaryLayer) {
                    map.removeLayer(boundaryLayer);
                    boundaryLayer = null;
                }
            }
        };

        init();
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
@require_password
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['GET', 'POST'])
@require_password
def upload_page():
    """upload page for data files"""
    status_lines = []
    status_lines.append(f"Data directory: {DATA_DIR}")
    status_lines.append(f"TILES_DIR: {TILES_DIR}")
    status_lines.append(f"IMAGE_PATH: {IMAGE_PATH}")
    status_lines.append("")
    status_lines.append(f"Data dir exists: {DATA_DIR.exists()}")
    status_lines.append(f"AB01 dir exists: {TILES_DIR.exists()}")
    status_lines.append(f"Image exists: {IMAGE_PATH.exists()}")
    status_lines.append(f"segmentation.npz exists: {(TILES_DIR / 'segmentation.npz').exists()}")
    status_lines.append(f"gene_index.json exists: {(TILES_DIR / 'gene_index.json').exists()}")
    status_lines.append(f"expression.zarr exists: {(TILES_DIR / 'expression.zarr').exists()}")

    return render_template_string(UPLOAD_TEMPLATE, status='\n'.join(status_lines))

@app.route('/upload/image', methods=['POST'])
@require_password
def upload_image():
    """upload TIFF image file"""
    try:
        file = request.files['file']
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        file.save(IMAGE_PATH)
        return f'Image uploaded successfully to {IMAGE_PATH}'
    except Exception as e:
        return f'Error uploading image: {str(e)}', 500

@app.route('/upload/npz', methods=['POST'])
@require_password
def upload_npz():
    """upload segmentation.npz file"""
    try:
        file = request.files['file']
        TILES_DIR.mkdir(parents=True, exist_ok=True)
        file.save(TILES_DIR / 'segmentation.npz')
        return f'NPZ file uploaded successfully to {TILES_DIR / "segmentation.npz"}'
    except Exception as e:
        return f'Error uploading NPZ: {str(e)}', 500

@app.route('/upload/json', methods=['POST'])
@require_password
def upload_json():
    """upload gene_index.json file"""
    try:
        file = request.files['file']
        TILES_DIR.mkdir(parents=True, exist_ok=True)
        file.save(TILES_DIR / 'gene_index.json')
        return f'JSON file uploaded successfully to {TILES_DIR / "gene_index.json"}'
    except Exception as e:
        return f'Error uploading JSON: {str(e)}', 500

@app.route('/upload/zarr', methods=['POST'])
@require_password
def upload_zarr():
    """upload zarr directory as zip file"""
    try:
        import zipfile
        file = request.files['file']
        TILES_DIR.mkdir(parents=True, exist_ok=True)

        # save zip temporarily
        zip_path = TILES_DIR / 'temp_zarr.zip'
        file.save(zip_path)

        # extract
        zarr_dir = TILES_DIR / 'expression.zarr'
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TILES_DIR)

        # remove zip
        zip_path.unlink()

        return f'Zarr directory uploaded successfully to {zarr_dir}'
    except Exception as e:
        return f'Error uploading Zarr: {str(e)}', 500

@app.route('/api/metadata', methods=['GET', 'POST'])
@require_password
def get_metadata():
    with open(TILES_DIR / 'gene_index.json') as f:
        gene_meta = json.load(f)
    return jsonify({'genes': gene_meta})

@app.route('/api/genes', methods=['GET', 'POST'])
@require_password
def get_genes():
    with open(TILES_DIR / 'gene_index.json') as f:
        data = json.load(f)
    return jsonify(data['genes'])

@app.route('/api/full_image', methods=['GET', 'POST'])
@require_password
def get_full_image_endpoint():
    img = get_full_image()
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=85)
    buf.seek(0)
    return send_file(buf, mimetype='image/jpeg')

@app.route('/api/expression/<gene>', methods=['GET', 'POST'])
@require_password
def get_expression(gene):
    try:
        zdata = load_zarr_expression()
        var_names = zdata['var_names'][:]
        obs_names = zdata['obs_names'][:]

        gene_idx = np.where(var_names == gene)[0]
        if len(gene_idx) == 0:
            return jsonify({'error': f'gene {gene} not found'}), 404

        expr = zdata['X'][:, gene_idx[0]]

        nonzero = np.where(expr > 0)[0]
        return jsonify({
            'stats': {
                'n_expressing': int(len(nonzero)),
                'total_cells': len(obs_names),
                'mean': float(np.mean(expr[nonzero])) if len(nonzero) > 0 else 0,
                'max': float(np.max(expr))
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# cache for generated overlays (keeps last 10 genes to conserve memory)
from functools import lru_cache as _lru_cache
OVERLAY_CACHE = {}
MAX_CACHE_SIZE = 10

def generate_expression_overlay(gene):
    """generate expression overlay image (cached)"""
    # check cache first
    if gene in OVERLAY_CACHE:
        print(f"Using cached overlay for {gene}")
        return OVERLAY_CACHE[gene]

    print(f"\n=== Creating expression overlay for {gene} ===")

    zdata = load_zarr_expression()
    seg = load_segmentation()
    print(f"Converting segmentation to dense...")
    seg_dense = seg.toarray()
    print(f"Loaded zarr and converted segmentation to dense: {seg_dense.shape}")

    var_names = zdata['var_names'][:]
    obs_names = zdata['obs_names'][:]

    gene_idx = np.where(var_names == gene)[0]
    if len(gene_idx) == 0:
        print(f"Gene {gene} not found")
        return None

    expr = zdata['X'][:, gene_idx[0]]
    print(f"Expression values: min={expr.min()}, max={expr.max()}, n_expressing={np.sum(expr > 0)}")

    print(f"Mapping expression to cells (vectorized)...")

    # create a lookup array: cell_id -> expression value
    max_cell_id = int(seg_dense.max())
    lookup = np.zeros(max_cell_id + 1, dtype=np.float32)

    # fill lookup table with expression values
    cell_ids = obs_names.astype(int)
    valid_mask = cell_ids <= max_cell_id
    lookup[cell_ids[valid_mask]] = expr[valid_mask]

    # vectorized mapping: use segmentation as index into lookup table
    expr_map = lookup[seg_dense]

    print(f"Expression map: min={expr_map.min()}, max={expr_map.max()}, n_nonzero={np.sum(expr_map > 0)}")

    # normalize and create image
    if np.max(expr_map) > 0:
        vmax = np.percentile(expr_map[expr_map > 0], 99)
        expr_norm = np.clip(expr_map / vmax * 255, 0, 255).astype(np.uint8)
        print(f"Normalized to vmax={vmax}, expr_norm range: {expr_norm.min()}-{expr_norm.max()}")
    else:
        expr_norm = expr_map.astype(np.uint8)
        print("No expression found, creating empty overlay")

    # apply colormap
    from matplotlib import cm
    cmap = cm.get_cmap('viridis')
    expr_rgba = cmap(expr_norm)
    expr_rgba[..., 3] = np.where(expr_norm > 0, 1.0, 0)  # alpha channel
    print(f"Applied viridis colormap, RGBA shape: {expr_rgba.shape}")

    # convert to image
    img = Image.fromarray((expr_rgba * 255).astype(np.uint8), mode='RGBA')
    print(f"Created PIL image: {img.size}")

    # downscale to match the displayed image
    new_size = (img.width // 2, img.height // 2)
    img = img.resize(new_size, Image.LANCZOS)
    print(f"Resized to: {img.size}")

    # save to bytes
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    png_bytes = buf.read()
    print(f"Saved PNG, size: {len(png_bytes)} bytes")

    # cache it (with size limit)
    if len(OVERLAY_CACHE) >= MAX_CACHE_SIZE:
        # remove oldest entry
        OVERLAY_CACHE.pop(next(iter(OVERLAY_CACHE)))
    OVERLAY_CACHE[gene] = png_bytes

    # cleanup temp arrays to free memory
    del seg_dense, lookup, expr_map, expr_norm, expr_rgba, img

    return png_bytes

@app.route('/api/expression_overlay/<gene>', methods=['GET', 'POST'])
@require_password
def get_expression_overlay(gene):
    try:
        png_bytes = generate_expression_overlay(gene)
        if png_bytes is None:
            return jsonify({'error': f'gene {gene} not found'}), 404

        buf = io.BytesIO(png_bytes)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        import traceback
        print(f"ERROR in expression overlay:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# cache for cell boundaries
BOUNDARY_CACHE = None

def generate_cell_boundaries():
    """generate cell boundary overlay image (cached)"""
    global BOUNDARY_CACHE

    if BOUNDARY_CACHE is not None:
        print("Using cached cell boundaries")
        return BOUNDARY_CACHE

    print("\n=== Creating cell boundaries ===")

    seg = load_segmentation()
    print(f"Converting segmentation to dense...")
    seg_dense = seg.toarray()
    print(f"Converted to dense: {seg_dense.shape}")

    # find boundaries using edge detection
    print(f"Detecting cell boundaries...")
    from scipy import ndimage

    # use sobel edge detection
    edges_y = ndimage.sobel(seg_dense.astype(float), axis=0)
    edges_x = ndimage.sobel(seg_dense.astype(float), axis=1)
    edges = np.hypot(edges_x, edges_y)

    # threshold to get clean boundaries
    edges = (edges > 0).astype(np.uint8) * 255
    print(f"Found {np.sum(edges > 0)} boundary pixels")

    # create RGBA image (yellow boundaries, transparent background)
    boundary_rgba = np.zeros((*edges.shape, 4), dtype=np.uint8)
    boundary_rgba[edges > 0] = [255, 255, 0, 255]  # yellow boundaries

    # convert to PIL image
    img = Image.fromarray(boundary_rgba, mode='RGBA')
    print(f"Created boundary image: {img.size}")

    # downscale to match displayed image
    new_size = (img.width // 2, img.height // 2)
    img = img.resize(new_size, Image.LANCZOS)
    print(f"Resized to: {img.size}")

    # save to bytes and cache
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    BOUNDARY_CACHE = buf.read()
    print(f"Saved PNG, size: {len(BOUNDARY_CACHE)} bytes")

    # cleanup temp arrays to free memory
    del seg_dense, edges, edges_x, edges_y, boundary_rgba, img

    return BOUNDARY_CACHE

@app.route('/api/cell_boundaries', methods=['GET', 'POST'])
@require_password
def get_cell_boundaries():
    """Generate cell boundary overlay image"""
    try:
        png_bytes = generate_cell_boundaries()
        buf = io.BytesIO(png_bytes)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        import traceback
        print(f"ERROR in cell boundaries:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("\n" + "="*60)
    print("Starting SIMPLIFIED Spatial Viewer")
    print("="*60)
    print("\n📡 Server is now accessible on your network!")
    print(f"\n🖥️  On this computer:")
    print(f"   http://localhost:5000")
    print(f"\n🌐 Share with others on same WiFi:")
    print(f"   http://{local_ip}:5000")
    print(f"\nPress Ctrl+C to stop\n")
    print("="*60 + "\n")

    app.run(debug=True, port=5000, host='0.0.0.0')
