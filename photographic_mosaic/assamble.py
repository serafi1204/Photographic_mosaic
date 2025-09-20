import numpy as np
import cv2
import os
import json
from tqdm import tqdm
import zipfile

INPUT_DIR = "source_originals"
OUTPUT_PATH = 'photographic_mosaic'
OUTPUT_DIR = OUTPUT_PATH + "/source"
OUTPUT_FILE = OUTPUT_PATH + '/photographic_mosaic.html'
OUTPUT_ZIP = 'photographic_mosaic.zip'


def generateMakedSource(source_file, mosaic_map, target, color_alpha=0.2, grayscale_alpha=0.5, save_path=INPUT_DIR):
    # load source
    sc = np.load(source_file)['data'].astype(np.uint8)


    # get size
    w, h = mosaic_map.shape
    sn, sw, sh, sch = sc.shape

    # resize target
    target = cv2.resize(target, (h*sh, w*sw)).astype(np.uint8)

    # generate
    for i in range(w): 
        for j in range(h):
            img = sc[mosaic_map[i, j]]
            partial_target = target[i*sw:(i+1)*sw, j*sh:(j+1)*sh]

            # BGR to LAB
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            partial_target_lab = cv2.cvtColor(partial_target, cv2.COLOR_BGR2LAB)

            # grayscale
            img_lab[:,:,0] = (img_lab[:,:,0] * (1 - grayscale_alpha) + partial_target_lab[:,:,0] * grayscale_alpha).astype(np.uint8)

            # color
            img_lab[:,:,1] = (img_lab[:,:,1] * (1 - color_alpha) + partial_target_lab[:,:,1] * color_alpha).astype(np.uint8)
            img_lab[:,:,2] = (img_lab[:,:,2] * (1 - color_alpha) + partial_target_lab[:,:,2] * color_alpha).astype(np.uint8)

            # LAB to BGR
            result = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
            
            # Save
            cv2.imwrite(f'{save_path}/{i*h+j}.jpg', result)
            
            print(f'Tiling ({i}, {j})...{((i+1)*w+(j+1))/(w*h)*100:0.1f}%\r', end='')
    print()

def prepare_image_levels(num_images, LEVEL_SCALES = [0.01, 0.1, 1.0] ,force_resize=True):
    NUM_LEVELS = len(LEVEL_SCALES)
    
    print("--- Step 1: Checking and Resizing Images ---")
    
    for n in tqdm(range(num_images), desc="Processing images"):
        original_path = os.path.join(INPUT_DIR, f"{n}.jpg")
        level_max_path = os.path.join(OUTPUT_DIR, f"({n})_l{NUM_LEVELS}.jpg")
        
        if not force_resize and os.path.exists(level_max_path):
            continue
        
        if not os.path.exists(original_path):
            continue
            
        image = cv2.imread(original_path)
        if image is None:
            continue
            
        h, w = image.shape[:2]
        for level in range(NUM_LEVELS):
            sf = LEVEL_SCALES[level]
            out_path = os.path.join(OUTPUT_DIR, f"({n})_l{level+1}.jpg")
            
            if level+1 == NUM_LEVELS and sf == 1.0:
                cv2.imwrite(out_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                t_w, t_h = int(w * sf), int(h * sf)
                t_img = cv2.resize(image, (t_w, t_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(out_path, t_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
    print("\nImage resizing step complete.")
    return True


def create_image_grid_html(COLS, ROWS, NUM_LEVELS, BASE_TILE_WIDTH):

    print("--- Step 2: Generating HTML File ---")
    if not os.path.isdir(OUTPUT_DIR):
        print("Error: Source directory '{OUTPUT_DIR}' not found.")
        return

    # Base64 인코딩 대신 파일 경로만 저장
    image_paths = {}
    total_images_processed = 0
    for n in tqdm(range(COLS * ROWS), desc="Getting image paths"):
        image_paths[n] = {}
        if os.path.exists(os.path.join(OUTPUT_DIR, f"({n})_l1.jpg")):
            total_images_processed += 1
            for level in range(1, NUM_LEVELS + 1):
                image_path = os.path.join('source', f"({n})_l{level}.jpg")
                # Base64 인코딩 대신 파일 경로만 저장
                image_paths[n][f"l{level}"] = os.path.join('source', f"({n})_l{level}.jpg").replace("\\", "/")

    json_paths = json.dumps(image_paths)
    
    aspect_ratio = 16/9
    try:
        sample_image = cv2.imread(os.path.join(INPUT_DIR, '(0).jpg'))
        if sample_image is not None:
            h, w, _ = sample_image.shape
            aspect_ratio = w/h
    except Exception:
        print("Warning: Could not determine image aspect ratio. Using 16/9 default.")

    final_html_template = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Canvas Image Grid</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background-color: #f0f0f0;
            overflow: hidden;
            cursor: grab;
        }}
        body.panning {{
            cursor: grabbing;
        }}
        canvas {{
            display: block;
            width: 100vw;
            height: 100vh;
        }}
    </style>
</head>
<body>
    <canvas id="imageCanvas"></canvas>

    <script>
        const canvas = document.getElementById('imageCanvas');
        const ctx = canvas.getContext('2d');
        let panX = 0, panY = 0;
        let scale = 1;
        let isPanning = false;
        let startX = 0, startY = 0;
        let lastPanX = 0, lastPanY = 0;
        
        const COLS = {COLS};
        const ROWS = {ROWS};
        const BASE_TILE_WIDTH = {BASE_TILE_WIDTH};
        const ASPECT_RATIO = {aspect_ratio};
        const NUM_LEVELS = {NUM_LEVELS};
        const LEVEL_SCALES = {{ 1: 0.25, 2: 0.6, 3: 1.0 }};
        const imagePaths = {json_paths};
        
        const images = new Map();
        const loadedLevels = new Map();
        let currentZoomLevel = 1;
        let animationFrameId = null;
        let initialScale = 1;

        function setInitialTransform() {{
            const gridWidth = COLS * BASE_TILE_WIDTH;
            const gridHeight = ROWS * (BASE_TILE_WIDTH / ASPECT_RATIO);
            const scaleX = canvas.width / gridWidth;
            const scaleY = canvas.height / gridHeight;
            initialScale = Math.min(scaleX, scaleY);
            scale = initialScale;
            panX = (canvas.width - gridWidth * scale) / 2;
            panY = (canvas.height - gridHeight * scale) / 2;
        }}
        
        function resizeCanvas() {{
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            setInitialTransform();
            render();
        }}
        
        function loadImages(visibleTiles, levelToLoad) {{
            for (const tileIndex of visibleTiles) {{
                if (imagePaths[tileIndex]) {{
                    const key = `${{tileIndex}}-${{levelToLoad}}`;
                    if (!images.has(key)) {{
                        const img = new Image();
                        img.onload = () => {{
                            images.set(key, img);
                            if (levelToLoad === currentZoomLevel) {{
                                render();
                            }}
                        }};
                        img.src = imagePaths[tileIndex][`l${{levelToLoad}}`];
                        images.set(key, 'loading');
                    }}
                }}
            }}
        }}

        function getVisibleTiles() {{
            const visibleTiles = [];
            const tileWidth = BASE_TILE_WIDTH * scale;
            const tileHeight = tileWidth / ASPECT_RATIO;
            const startCol = Math.max(0, Math.floor((-panX) / tileWidth) - 1);
            const endCol = Math.min(COLS, Math.ceil((-panX + canvas.width) / tileWidth) + 1);
            const startRow = Math.max(0, Math.floor((-panY) / tileHeight) - 1);
            const endRow = Math.min(ROWS, Math.ceil((-panY + canvas.height) / tileHeight) + 1);

            for (let r = startRow; r < endRow; r++) {{
                for (let c = startCol; c < endCol; c++) {{
                    visibleTiles.push(r * COLS + c);
                }}
            }}
            return visibleTiles;
        }}

        function clampPan() {{
            const gridWidth = COLS * BASE_TILE_WIDTH * scale;
            const gridHeight = ROWS * (BASE_TILE_WIDTH / ASPECT_RATIO) * scale;
            
            const minX = (gridWidth > canvas.width) ? canvas.width - gridWidth : (canvas.width - gridWidth) / 2;
            const maxX = (gridWidth > canvas.width) ? 0 : (canvas.width - gridWidth) / 2;
            const minY = (gridHeight > canvas.height) ? canvas.height - gridHeight : (canvas.height - gridHeight) / 2;
            const maxY = (gridHeight > canvas.height) ? 0 : (canvas.height - gridHeight) / 2;
            
            panX = Math.max(minX, Math.min(panX, maxX));
            panY = Math.max(minY, Math.min(panY, maxY));
        }}

        function render() {{
            if (animationFrameId) return;
            animationFrameId = requestAnimationFrame(() => {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                const visibleTiles = getVisibleTiles();
                const tileWidth = BASE_TILE_WIDTH * scale;
                const tileHeight = tileWidth / ASPECT_RATIO;
                
                loadImages(visibleTiles, isPanning ? 1 : currentZoomLevel);
                
                for (const tileIndex of visibleTiles) {{
                    const row = Math.floor(tileIndex / COLS);
                    const col = tileIndex % COLS;
                    const x = panX + col * tileWidth;
                    const y = panY + row * tileHeight;

                    const renderLevel = isPanning ? 1 : currentZoomLevel;
                    const key = `${{tileIndex}}-${{renderLevel}}`;
                    
                    let img = images.get(key);
                    if (!img || img === 'loading') {{
                        img = images.get(`${{tileIndex}}-1`);
                    }}
                    
                    if (img && img !== 'loading') {{
                        ctx.drawImage(img, x, y, tileWidth, tileHeight);
                    }}
                }}
                animationFrameId = null;
            }});
        }}

        canvas.addEventListener('mousedown', (e) => {{
            e.preventDefault();
            isPanning = true;
            startX = e.clientX;
            startY = e.clientY;
            lastPanX = panX;
            lastPanY = panY;
            document.body.classList.add('panning');
            render();
        }});

        window.addEventListener('mouseup', () => {{
            isPanning = false;
            document.body.classList.remove('panning');
            render();
        }});

        window.addEventListener('mousemove', (e) => {{
            if (!isPanning) return;
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            panX = lastPanX + dx;
            panY = lastPanY + dy;
            
            clampPan();
            render();
        }});

        canvas.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const mouseX = e.clientX;
            const mouseY = e.clientY;
            const oldScale = scale;
            
            if (e.deltaY < 0) {{
                scale *= 1.1;
            }} else {{
                scale /= 1.1;
            }}
            
            scale = Math.max(initialScale, Math.min(scale, 10));

            panX = mouseX - (mouseX - panX) * (scale / oldScale);
            panY = mouseY - (mouseY - panY) * (scale / oldScale);

            clampPan();

            const newZoomLevel = Math.max(1, Math.min(NUM_LEVELS, Math.floor(scale / initialScale * (NUM_LEVELS / 2))));
            if (newZoomLevel !== currentZoomLevel) {{
                currentZoomLevel = newZoomLevel;
            }}
            render();
        }});

        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
    </script>
</body>
</html>
'''

    final_html = final_html_template.format(
        COLS=COLS, ROWS=ROWS, NUM_LEVELS=NUM_LEVELS, BASE_TILE_WIDTH=BASE_TILE_WIDTH, aspect_ratio=aspect_ratio,
        json_paths=json_paths
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f: f.write(final_html)
    print(f"\nSuccessfully generated '{OUTPUT_FILE}'. You can now open this file in your web browser.")

def zip():
    print('zip...')
    zip_file = zipfile.ZipFile(OUTPUT_ZIP, "w")
    for (path, dir, files) in os.walk(OUTPUT_PATH):
        for file in files:
            zip_file.write(os.path.join(path, file), compress_type=zipfile.ZIP_DEFLATED)

    zip_file.close()

def assemble(source_file, mosaic_map, target, color_alpha=0.1, grayscale_alpha=0.2, LEVEL_SCALES = [0.01, 0.1, 1.0], BASE_TILE_WIDTH=720):
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    
    w, h = mosaic_map.shape
    num_images = w*h

    generateMakedSource(source_file, mosaic_map, target, color_alpha=color_alpha, grayscale_alpha=grayscale_alpha)
    prepare_image_levels(num_images, LEVEL_SCALES)
    create_image_grid_html(w, h, len(LEVEL_SCALES), BASE_TILE_WIDTH)
    zip()

    return OUTPUT_PATH