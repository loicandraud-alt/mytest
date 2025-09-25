const statusEl = document.getElementById('status');
const runButton = document.getElementById('run-button');
const fileInput = document.getElementById('image-input');
const dilationInput = document.getElementById('dilation-input');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const resultsContainer = document.getElementById('results');

let imageLoadPromise = null;
let currentImageURL = null;

const cvReady = new Promise((resolve) => {
  if (typeof cv !== 'undefined' && cv.Mat) {
    resolve();
  } else {
    const previous = cv.onRuntimeInitialized;
    cv.onRuntimeInitialized = () => {
      if (typeof previous === 'function') {
        previous();
      }
      resolve();
    };
  }
});

function updateStatus(message, type = 'info') {
  statusEl.textContent = message;
  statusEl.style.color = type === 'error' ? '#dc2626' : '#1d4ed8';
}

function resetResults() {
  while (resultsContainer.firstChild) {
    resultsContainer.removeChild(resultsContainer.firstChild);
  }
}

function revokeCurrentImageURL() {
  if (currentImageURL) {
    URL.revokeObjectURL(currentImageURL);
    currentImageURL = null;
  }
}

fileInput.addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  resetResults();
  updateStatus('');

  if (!file) {
    previewContainer.hidden = true;
    revokeCurrentImageURL();
    imageLoadPromise = null;
    return;
  }

  imageLoadPromise = new Promise((resolve, reject) => {
    revokeCurrentImageURL();
    currentImageURL = URL.createObjectURL(file);
    previewImage.onload = () => {
      previewContainer.hidden = false;
      resolve();
    };
    previewImage.onerror = () => {
      updateStatus("Impossible de charger l'image.", 'error');
      previewContainer.hidden = true;
      reject(new Error('Image load error'));
    };
    previewImage.src = currentImageURL;
  });
});

runButton.addEventListener('click', async () => {
  if (!imageLoadPromise) {
    updateStatus('Sélectionnez d\'abord une image.', 'error');
    return;
  }

  runButton.disabled = true;
  updateStatus('Traitement en cours…');

  try {
    await imageLoadPromise;
    await cvReady;
    const dilation = Math.max(1, Math.min(15, Number.parseInt(dilationInput.value, 10) || 4));
    const results = await drawFile(previewImage, dilation, cv.RETR_CCOMP);
    displayResults(results);
    updateStatus(`Analyse terminée : ${results.length} images générées.`);
  } catch (error) {
    console.error(error);
    updateStatus('Une erreur est survenue pendant le traitement.', 'error');
  } finally {
    runButton.disabled = false;
  }
});

cvReady.then(() => {
  updateStatus('OpenCV.js est chargé. Importez une image pour démarrer.');
  runButton.disabled = false;
});

async function drawFile(imageElement, dilation, mode) {
  const src = cv.imread(imageElement);
  const { image, edges } = preprocessImage(src, dilation);
  src.delete();

  const contours = extractContours(edges, mode);
  const textures = loadTextures();

  const overlays = processContours(image, contours, textures);
  const colorZones = buildColorZones(image, contours);

  const original = image.clone();

  const edgesColor = new cv.Mat();
  cv.cvtColor(edges, edgesColor, cv.COLOR_GRAY2BGR);

  edges.delete();
  image.delete();

  const results = [
    { label: 'Image originale', mat: original },
    { label: 'Contours dilatés', mat: edgesColor },
    { label: 'Remplissage texturé', mat: overlays.texturedImage },
    { label: 'Quadrilatères et bords', mat: overlays.backgroundWithQuads },
    { label: 'Points caractéristiques', mat: overlays.pointsOverlay },
    { label: 'Approximation polygonale', mat: overlays.approxOverlay },
    { label: 'Contours accentués', mat: overlays.hollowOverlay },
    { label: 'Concavités simulées', mat: overlays.concavityOverlay },
    { label: 'Zones colorées', mat: colorZones }
  ];

  contours.forEach((cnt) => cnt.delete());
  textures.forEach((tex) => tex.delete());

  return results;
}

function preprocessImage(src, dilation) {
  const bgr = new cv.Mat();
  cv.cvtColor(src, bgr, cv.COLOR_RGBA2BGR);

  const grayBoosted = boostImageGray(bgr);

  const edges = new cv.Mat();
  cv.Canny(grayBoosted, edges, 1, 150);

  const kernelSize = Math.max(1, dilation);
  const kernel = cv.Mat.ones(kernelSize, kernelSize, cv.CV_8UC1);
  const edgesDilated = new cv.Mat();
  cv.dilate(edges, edgesDilated, kernel);

  grayBoosted.delete();
  edges.delete();
  kernel.delete();

  const image = bgr.clone();
  bgr.delete();

  return { image, edges: edgesDilated };
}

function boostImageGray(imageBGR) {
  const hsv = new cv.Mat();
  cv.cvtColor(imageBGR, hsv, cv.COLOR_BGR2HSV);

  const channels = new cv.MatVector();
  cv.split(hsv, channels);
  const h = channels.get(0);
  const s = channels.get(1);
  const v = channels.get(2);

  cv.convertScaleAbs(s, s, 1.5, 0);
  const saturationMax = new cv.Mat(s.rows, s.cols, s.type(), new cv.Scalar(255));
  cv.min(s, saturationMax, s);

  const merged = new cv.Mat();
  const mergedVec = new cv.MatVector();
  mergedVec.push_back(h);
  mergedVec.push_back(s);
  mergedVec.push_back(v);
  cv.merge(mergedVec, merged);

  const boosted = new cv.Mat();
  cv.cvtColor(merged, boosted, cv.COLOR_HSV2BGR);

  const gray = new cv.Mat();
  cv.cvtColor(boosted, gray, cv.COLOR_BGR2GRAY);

  hsv.delete();
  channels.delete();
  h.delete();
  s.delete();
  v.delete();
  saturationMax.delete();
  merged.delete();
  mergedVec.delete();
  boosted.delete();

  return gray;
}

function extractContours(edgeImage, mode) {
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(edgeImage, contours, hierarchy, mode, cv.CHAIN_APPROX_SIMPLE);
  hierarchy.delete();

  const contourList = [];
  for (let i = 0; i < contours.size(); i += 1) {
    const cnt = contours.get(i);
    contourList.push(cnt);
  }
  contours.delete();
  return contourList;
}

function processContours(image, contours, textures) {
  const texturedImage = image.clone();
  const backgroundWithQuads = image.clone();
  const pointsOverlay = image.clone();
  const approxOverlay = image.clone();
  const hollowOverlay = image.clone();
  const concavityOverlay = image.clone();

  contours.forEach((contour) => {
    const area = Math.abs(cv.contourArea(contour));
    if (area < 200) {
      return;
    }

    const contourVec = new cv.MatVector();
    contourVec.push_back(contour);

    const mask = cv.Mat.zeros(image.rows, image.cols, cv.CV_8UC1);
    cv.drawContours(mask, contourVec, 0, new cv.Scalar(255, 255, 255, 255), -1);

    const rect = cv.boundingRect(contour);
    cv.rectangle(
      backgroundWithQuads,
      new cv.Point(rect.x, rect.y),
      new cv.Point(rect.x + rect.width, rect.y + rect.height),
      new cv.Scalar(0, 0, 255, 255),
      2
    );

    const approx = new cv.Mat();
    const epsilon = 0.02 * cv.arcLength(contour, true);
    cv.approxPolyDP(contour, approx, epsilon, true);
    const approxVec = new cv.MatVector();
    approxVec.push_back(approx);
    cv.polylines(approxOverlay, approxVec, true, new cv.Scalar(0, 255, 0, 255), 2);
    approx.delete();
    approxVec.delete();

    const texture = textures[Math.floor(Math.random() * textures.length)];
    const tiled = tileTexture(texture, Math.max(rect.width, 1), Math.max(rect.height, 1));
    const roiDest = texturedImage.roi(rect);
    const roiMask = mask.roi(rect);
    tiled.copyTo(roiDest, roiMask);
    roiDest.delete();
    roiMask.delete();
    tiled.delete();

    const moments = cv.moments(contour, false);
    let cx = rect.x + rect.width / 2;
    let cy = rect.y + rect.height / 2;
    if (Math.abs(moments.m00) > 1e-5) {
      cx = moments.m10 / moments.m00;
      cy = moments.m01 / moments.m00;
    }
    const center = new cv.Point(Math.round(cx), Math.round(cy));
    cv.circle(pointsOverlay, center, 6, new cv.Scalar(0, 0, 255, 255), -1);

    cv.drawContours(hollowOverlay, contourVec, 0, new cv.Scalar(0, 255, 255, 255), 2);
    cv.drawContours(concavityOverlay, contourVec, 0, new cv.Scalar(0, 128, 255, 255), 2);

    contourVec.delete();
    mask.delete();
  });

  return {
    texturedImage,
    backgroundWithQuads,
    pointsOverlay,
    approxOverlay,
    hollowOverlay,
    concavityOverlay
  };
}

function buildColorZones(image, contours) {
  const colorZones = cv.Mat.zeros(image.rows, image.cols, cv.CV_8UC3);

  contours.forEach((contour, index) => {
    const area = Math.abs(cv.contourArea(contour));
    if (area < 200) {
      return;
    }
    const color = new cv.Scalar(
      Math.floor(Math.random() * 180) + 50,
      Math.floor(Math.random() * 180) + 50,
      Math.floor(Math.random() * 180) + 50,
      255
    );
    const contourVec = new cv.MatVector();
    contourVec.push_back(contour);
    cv.drawContours(colorZones, contourVec, 0, color, -1);

    const moments = cv.moments(contour, false);
    let cx;
    let cy;
    if (Math.abs(moments.m00) > 1e-5) {
      cx = moments.m10 / moments.m00;
      cy = moments.m01 / moments.m00;
    } else {
      const rect = cv.boundingRect(contour);
      cx = rect.x + rect.width / 2;
      cy = rect.y + rect.height / 2;
    }
    cv.putText(
      colorZones,
      String(index),
      new cv.Point(Math.round(cx), Math.round(cy)),
      cv.FONT_HERSHEY_SIMPLEX,
      0.6,
      new cv.Scalar(255, 255, 255, 255),
      2
    );
    contourVec.delete();
  });

  return colorZones;
}

function tileTexture(texture, width, height) {
  if (width <= 0 || height <= 0) {
    throw new Error('Dimensions cibles invalides pour la texture.');
  }
  const result = new cv.Mat(height, width, texture.type());
  for (let y = 0; y < height; y += texture.rows) {
    const tileHeight = Math.min(texture.rows, height - y);
    for (let x = 0; x < width; x += texture.cols) {
      const tileWidth = Math.min(texture.cols, width - x);
      const roiResult = result.roi(new cv.Rect(x, y, tileWidth, tileHeight));
      const roiTexture = texture.roi(new cv.Rect(0, 0, tileWidth, tileHeight));
      roiTexture.copyTo(roiResult);
      roiResult.delete();
      roiTexture.delete();
    }
  }
  return result;
}

function loadTextures() {
  return [
    createCheckerTexture(64, 16, new cv.Scalar(60, 40, 200, 255), new cv.Scalar(200, 200, 60, 255)),
    createStripesTexture(64, 12, new cv.Scalar(40, 120, 220, 255), new cv.Scalar(220, 100, 80, 255)),
    createNoiseTexture(72, [120, 80, 60])
  ];
}

function createCheckerTexture(size, blockSize, colorA, colorB) {
  const mat = new cv.Mat(size, size, cv.CV_8UC3);
  for (let y = 0; y < size; y += blockSize) {
    for (let x = 0; x < size; x += blockSize) {
      const useA = ((x / blockSize) + (y / blockSize)) % 2 === 0;
      cv.rectangle(
        mat,
        new cv.Point(x, y),
        new cv.Point(Math.min(x + blockSize, size), Math.min(y + blockSize, size)),
        useA ? colorA : colorB,
        cv.FILLED
      );
    }
  }
  return mat;
}

function createStripesTexture(size, stripeWidth, colorA, colorB) {
  const mat = new cv.Mat(size, size, cv.CV_8UC3);
  for (let x = 0; x < size; x += stripeWidth) {
    const useA = Math.floor(x / stripeWidth) % 2 === 0;
    cv.rectangle(
      mat,
      new cv.Point(x, 0),
      new cv.Point(Math.min(x + stripeWidth, size), size),
      useA ? colorA : colorB,
      cv.FILLED
    );
  }
  return mat;
}

function createNoiseTexture(size, baseColor) {
  const mat = new cv.Mat(size, size, cv.CV_8UC3);
  const data = mat.data;
  for (let i = 0; i < data.length; i += 3) {
    const noise = Math.floor(Math.random() * 60);
    data[i] = Math.min(255, baseColor[0] + noise);
    data[i + 1] = Math.min(255, baseColor[1] + noise);
    data[i + 2] = Math.min(255, baseColor[2] + noise);
  }
  return mat;
}

function displayResults(results) {
  resetResults();
  results.forEach(({ label, mat }) => {
    const item = document.createElement('article');
    item.className = 'result-item';

    const title = document.createElement('h3');
    title.textContent = label;
    item.appendChild(title);

    const canvas = document.createElement('canvas');
    canvas.width = mat.cols;
    canvas.height = mat.rows;
    cv.imshow(canvas, mat);
    item.appendChild(canvas);

    resultsContainer.appendChild(item);

    mat.delete();
  });
}