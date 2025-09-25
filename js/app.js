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
    if (typeof error === 'number' && cv?.exceptionFromPtr) {
      const cvError = cv.exceptionFromPtr(error);
      console.error(cvError?.msg || error);
    } else {
      console.error(error);
    }
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

  const contours = floodfillExtractContours(edges, mode);
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

function floodfillExtractContours(edgeImage, mode) {
  const img = edgeImage.clone();
  const h = img.rows;
  const w = img.cols;
  const totalPixels = h * w;
  const contourAreaThreshold = 0.002 * totalPixels;
  const contours = [];

  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      if (img.ucharPtr(y, x)[0] !== 0) {
        continue;
      }

      const floodMask = cv.Mat.zeros(h + 2, w + 2, cv.CV_8UC1);
      cv.floodFill(
        img,
        floodMask,
        new cv.Point(x, y),
        new cv.Scalar(255),
        new cv.Rect(),
        new cv.Scalar(0),
        new cv.Scalar(0),
        4
      );

      const filledRoi = floodMask.roi(new cv.Rect(1, 1, w, h));
      const filledArea = filledRoi.clone();
      filledArea.convertTo(filledArea, cv.CV_8UC1, 255);

      const surfacePixels = cv.countNonZero(filledArea);
      if (
        surfacePixels > contourAreaThreshold &&
        !filledAreaTouchesTop(filledArea) &&
        !filledAreaTouchesBottom(filledArea)
      ) {
        const cnts = new cv.MatVector();
        const hierarchy = new cv.Mat();
        cv.findContours(filledArea, cnts, hierarchy, mode, cv.CHAIN_APPROX_SIMPLE);
        hierarchy.delete();

        for (let i = 0; i < cnts.size(); i += 1) {
          const cnt = cnts.get(i);
          contours.push(cnt.clone());
          cnt.delete();
        }
        cnts.delete();
      }

      filledArea.delete();
      filledRoi.delete();
      floodMask.delete();
    }
  }

  img.delete();
  return contours;
}

function processContours(image, contours, textures) {
  let texturedImage = image.clone();
  const backgroundWithQuads = image.clone();
  const pointsOverlay = image.clone();
  const approxOverlay = image.clone();
  const hollowOverlay = image.clone();
  const concavityOverlay = image.clone();

  const inclusionMap = checkContoursInside(contours);
  let processedCount = 0;

  contours.forEach((contour, index) => {
    if (!contour || contour.rows === 0) {
      return;
    }

    const area = contourArea(contour);
    if (area < 200) {
      return;
    }

    const parents = inclusionMap[index]?.parents ?? [];
    if (parents.length > 0) {
      return;
    }

    if (processedCount >= 100) {
      return;
    }
    processedCount += 1;

    const detection = detectLargestHollowParallelepiped(
      contour,
      { rows: image.rows, cols: image.cols }
    );

    if (detection && detection.best) {
      const { best } = detection;
      if (best.box) {
        const boxMat = pointArrayToMat(best.box.map((pt) => [Math.round(pt[0]), Math.round(pt[1])]));
        const boxVec = new cv.MatVector();
        boxVec.push_back(boxMat);
        cv.drawContours(hollowOverlay, boxVec, 0, new cv.Scalar(0, 255, 255, 255), 3);
        boxVec.delete();
        boxMat.delete();
      }

      if (best.diffContour) {
        const diffVec = new cv.MatVector();
        diffVec.push_back(best.diffContour);
        cv.drawContours(hollowOverlay, diffVec, 0, new cv.Scalar(255, 0, 255, 255), 2);
        diffVec.delete();
        best.diffContour.delete();
        best.diffContour = null;
      }

      if (best.rect && best.area) {
        const areaText = `${Math.round(best.area)} px²`;
        const [cx, cy] = best.rect.center;
        const position = new cv.Point(Math.round(cx), Math.round(cy));
        cv.putText(
          hollowOverlay,
          areaText,
          position,
          cv.FONT_HERSHEY_SIMPLEX,
          0.6,
          new cv.Scalar(0, 0, 0, 255),
          3,
          cv.LINE_AA
        );
        cv.putText(
          hollowOverlay,
          areaText,
          position,
          cv.FONT_HERSHEY_SIMPLEX,
          0.6,
          new cv.Scalar(0, 255, 255, 255),
          1,
          cv.LINE_AA
        );
      }
    }

    const concavities = findConcavities(contour);
    concavities.forEach((concavity) => {
      if (!concavity || concavity.rows < 3) {
        if (concavity) {
          concavity.delete();
        }
        return;
      }

      const concavityVec = new cv.MatVector();
      concavityVec.push_back(concavity);
      cv.polylines(concavityOverlay, concavityVec, true, new cv.Scalar(0, 255, 255, 255), 2);
      concavityVec.delete();

      const quad = largestQuadrilateralInConcavity(concavity);
      if (quad) {
        const quadMat = pointArrayToMat(quad.map((pt) => [Math.round(pt[0]), Math.round(pt[1])]));
        const quadVec = new cv.MatVector();
        quadVec.push_back(quadMat);
        cv.polylines(concavityOverlay, quadVec, true, new cv.Scalar(0, 0, 255, 255), 2);
        quadVec.delete();
        quadMat.delete();
      }

      concavity.delete();
    });

    const { points, approx } = findPointsFromContour(contour);
    let quadrilateralPoints = null;

    if (approx) {
      const approxVec = new cv.MatVector();
      approxVec.push_back(approx);
      cv.polylines(approxOverlay, approxVec, true, new cv.Scalar(0, 255, 0, 255), 3);
      approxVec.delete();
      approx.delete();
    }

    if (points) {
      const [pt11, pt12, pt21, pt22] = points;
      [pt11, pt12, pt21, pt22].forEach((pt) => {
        cv.circle(
          pointsOverlay,
          new cv.Point(Math.round(pt[0]), Math.round(pt[1])),
          8,
          new cv.Scalar(255, 0, 0, 255),
          -1
        );
      });

      try {
        quadrilateralPoints = quadrilateralFromLines([pt11, pt12], [pt21, pt22]);
      } catch (error) {
        quadrilateralPoints = null;
      }

      if (quadrilateralPoints && !isValidQuadrilateral(quadrilateralPoints)) {
        quadrilateralPoints = null;
      }

      if (quadrilateralPoints) {
        const quadMat = pointArrayToMat(
          quadrilateralPoints.map((pt) => [Math.round(pt[0]), Math.round(pt[1])])
        );
        const quadVec = new cv.MatVector();
        quadVec.push_back(quadMat);
        cv.polylines(backgroundWithQuads, quadVec, true, new cv.Scalar(0, 0, 255, 255), 3);
        quadVec.delete();
        quadMat.delete();
      }
    }

    const wallMask = buildContourMask(contour, contours, image);
    const chosenTexture = textures[Math.floor(Math.random() * textures.length)];
    const dilated = dilateContour(contour, image, 3);

    if (!dilated) {
      wallMask.delete();
      return;
    }

    const rect = cv.boundingRect(dilated);
    const overlayPoints = quadrilateralPoints && isValidQuadrilateral(quadrilateralPoints)
      ? [
          quadrilateralPoints[0],
          quadrilateralPoints[3],
          quadrilateralPoints[2],
          quadrilateralPoints[1]
        ]
      : null;

    if (overlayPoints) {
      try {
        const tiled = tileTexture(chosenTexture, Math.max(rect.width, 1), Math.max(rect.height, 1));
        const projected = projectTexture(texturedImage, tiled, overlayPoints, wallMask);
        tiled.delete();
        wallMask.delete();
        dilated.delete();
        texturedImage.delete();
        texturedImage = projected;
        return;
      } catch (error) {
        console.warn('Projection perspective échouée, utilisation du remplissage classique:', error);
      }
    }

    const tiled = tileTexture(chosenTexture, Math.max(rect.width, 1), Math.max(rect.height, 1));
    const mask = new cv.Mat.zeros(rect.height, rect.width, cv.CV_8UC1);
    const shifted = translateContour(dilated, -rect.x, -rect.y);
    const contourVec = new cv.MatVector();
    contourVec.push_back(shifted);
    cv.drawContours(mask, contourVec, 0, new cv.Scalar(255, 255, 255, 255), cv.FILLED);
    contourVec.delete();
    shifted.delete();

    const texturedRegion = new cv.Mat();
    cv.bitwise_and(tiled, tiled, texturedRegion, mask);

    const roi = texturedImage.roi(rect);
    const invMask = new cv.Mat();
    cv.bitwise_not(mask, invMask);

    const roiBg = new cv.Mat();
    cv.bitwise_and(roi, roiBg, invMask);

    const combined = new cv.Mat();
    cv.add(roiBg, texturedRegion, combined);
    combined.copyTo(roi);

    roi.delete();
    invMask.delete();
    roiBg.delete();
    combined.delete();
    texturedRegion.delete();
    tiled.delete();
    mask.delete();
    wallMask.delete();
    dilated.delete();
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
    const area = contourArea(contour);
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
    cv.drawContours(colorZones, contourVec, 0, color, cv.FILLED);
    contourVec.delete();

    const centroid = contourCentroid(contour);
    const cx = Math.round(centroid[0]);
    const cy = Math.round(centroid[1]);
    cv.putText(
      colorZones,
      String(index),
      new cv.Point(cx, cy),
      cv.FONT_HERSHEY_SIMPLEX,
      0.6,
      new cv.Scalar(255, 255, 255, 255),
      2
    );
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

function filledAreaTouchesTop(mat) {
  if (!mat || mat.rows === 0) {
    return false;
  }
  for (let x = 0; x < mat.cols; x += 1) {
    if (mat.ucharPtr(0, x)[0] !== 0) {
      return true;
    }
  }
  return false;
}

function filledAreaTouchesBottom(mat) {
  if (!mat || mat.rows === 0) {
    return false;
  }
  const lastRow = mat.rows - 1;
  for (let x = 0; x < mat.cols; x += 1) {
    if (mat.ucharPtr(lastRow, x)[0] !== 0) {
      return true;
    }
  }
  return false;
}

function contourArea(contour) {
  if (!contour || contour.rows === 0) {
    return 0;
  }
  return Math.abs(cv.contourArea(contour));
}

function contourCentroid(contour) {
  const moments = cv.moments(contour, false);
  if (Math.abs(moments.m00) > 1e-5) {
    return [moments.m10 / moments.m00, moments.m01 / moments.m00];
  }
  const points = matToPointArray(contour);
  if (points.length > 0) {
    return [points[0][0], points[0][1]];
  }
  return [0, 0];
}

function matToPointArray(mat) {
  const points = [];
  if (!mat) {
    return points;
  }
  const type = mat.type();
  const isInt = type === cv.CV_32SC2;
  for (let i = 0; i < mat.rows; i += 1) {
    if (isInt) {
      const ptr = mat.intPtr(i, 0);
      points.push([ptr[0], ptr[1]]);
    } else {
      const ptr = mat.floatPtr(i, 0);
      points.push([ptr[0], ptr[1]]);
    }
  }
  return points;
}

function pointArrayToMat(points, type = cv.CV_32SC2) {
  if (!points || points.length === 0) {
    return new cv.Mat();
  }
  const mat = new cv.Mat(points.length, 1, type);
  const isInt = type === cv.CV_32SC2;
  for (let i = 0; i < points.length; i += 1) {
    if (isInt) {
      const ptr = mat.intPtr(i, 0);
      ptr[0] = Math.round(points[i][0]);
      ptr[1] = Math.round(points[i][1]);
    } else {
      const ptr = mat.floatPtr(i, 0);
      ptr[0] = points[i][0];
      ptr[1] = points[i][1];
    }
  }
  return mat;
}

function translateContour(contour, dx, dy) {
  const points = matToPointArray(contour).map((pt) => [pt[0] + dx, pt[1] + dy]);
  return pointArrayToMat(points, contour.type());
}

function checkContoursInside(contours) {
  return contours.map((contour, index) => {
    if (!contour || contour.rows === 0) {
      console.log(`Contour ${index} est vide ou non défini.`);
      return { index, parents: [] };
    }

    const parents = [];
    const points = matToPointArray(contour);

    contours.forEach((other, otherIndex) => {
      if (otherIndex === index || !other || other.rows === 0) {
        return;
      }
      const inside = points.every((pt) => (
        cv.pointPolygonTest(other, new cv.Point(pt[0], pt[1]), false) >= 0
      ));
      if (inside) {
        parents.push(otherIndex);
      }
    });

    if (parents.length > 0) {
      console.log(`Contour ${index} est entouré par les contours ${parents}.`);
    } else {
      console.log(`Contour ${index} n'est entouré par aucun autre contour.`);
    }

    return { index, parents };
  });
}

function buildContourMask(targetContour, allContours, image) {
  const mask = cv.Mat.zeros(image.rows, image.cols, cv.CV_8UC1);
  const vec = new cv.MatVector();
  vec.push_back(targetContour);
  cv.drawContours(mask, vec, 0, new cv.Scalar(255, 255, 255, 255), cv.FILLED);
  vec.delete();

  const exclusion = cv.Mat.zeros(image.rows, image.cols, cv.CV_8UC1);

  allContours.forEach((contour) => {
    if (!contour || contour.rows === 0 || contour === targetContour) {
      return;
    }
    const centroid = contourCentroid(contour);
    if (cv.pointPolygonTest(targetContour, new cv.Point(centroid[0], centroid[1]), false) >= 0) {
      const vecExclusion = new cv.MatVector();
      vecExclusion.push_back(contour);
      cv.drawContours(exclusion, vecExclusion, 0, new cv.Scalar(255, 255, 255, 255), cv.FILLED);
      vecExclusion.delete();
    }
  });

  if (cv.countNonZero(exclusion) === 0) {
    exclusion.delete();
    return mask;
  }

  const inverted = new cv.Mat();
  cv.bitwise_not(exclusion, inverted);
  const filtered = new cv.Mat();
  cv.bitwise_and(mask, inverted, filtered);
  mask.delete();
  exclusion.delete();
  inverted.delete();
  return filtered;
}

function dilateContour(contour, image, dilationPx = 4) {
  const mask = cv.Mat.zeros(image.rows, image.cols, cv.CV_8UC1);
  const vec = new cv.MatVector();
  vec.push_back(contour);
  cv.drawContours(mask, vec, 0, new cv.Scalar(255, 255, 255, 255), cv.FILLED);
  vec.delete();

  const kernelSize = Math.max(1, 2 * dilationPx + 1);
  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kernelSize, kernelSize));
  const dilatedMask = new cv.Mat();
  cv.dilate(mask, dilatedMask, kernel);
  mask.delete();
  kernel.delete();

  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(dilatedMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  hierarchy.delete();
  dilatedMask.delete();

  let result = null;
  for (let i = 0; i < contours.size(); i += 1) {
    const cnt = contours.get(i);
    if (!result || contourArea(cnt) > contourArea(result)) {
      if (result) {
        result.delete();
      }
      result = cnt.clone();
    }
    cnt.delete();
  }
  contours.delete();
  return result;
}

function contourSegment(contour, startIdx, endIdx) {
  if (!contour || contour.rows === 0) {
    return new cv.Mat();
  }
  const n = contour.rows;
  const points = [];
  let idx = ((startIdx % n) + n) % n;
  const end = ((endIdx % n) + n) % n;

  while (true) {
    const ptr = contour.intPtr(idx, 0);
    points.push([ptr[0], ptr[1]]);
    if (idx === end) {
      break;
    }
    idx = (idx + 1) % n;
    if (idx === ((startIdx % n) + n) % n) {
      break;
    }
  }

  if (points.length < 3) {
    return new cv.Mat();
  }
  return pointArrayToMat(points);
}

function findConcavities(contour, minDepth = 5.0, minArea = 25.0) {
  if (!contour || contour.rows < 3) {
    return [];
  }

  const hull = new cv.Mat();
  cv.convexHull(contour, hull, false, false);
  if (!hull || hull.rows < 3) {
    if (hull) {
      hull.delete();
    }
    return [];
  }

  const defects = new cv.Mat();
  cv.convexityDefects(contour, hull, defects);
  hull.delete();

  if (!defects || defects.rows === 0) {
    if (defects) {
      defects.delete();
    }
    return [];
  }

  const concavities = [];
  const depthThreshold = Math.max(0, minDepth) * 256.0;

  for (let i = 0; i < defects.rows; i += 1) {
    const ptr = defects.intPtr(i, 0);
    const startIdx = ptr[0];
    const endIdx = ptr[1];
    const farIdx = ptr[2];
    const depth = ptr[3];

    if (depth < depthThreshold) {
      continue;
    }

    let segment = contourSegment(contour, startIdx, endIdx);
    if (!segment || segment.rows === 0) {
      if (segment) {
        segment.delete();
      }
      continue;
    }

    const farPtr = contour.intPtr(farIdx % contour.rows, 0);
    const farPoint = [farPtr[0], farPtr[1]];
    const points = matToPointArray(segment);
    const hasFarPoint = points.some((pt) => pt[0] === farPoint[0] && pt[1] === farPoint[1]);
    if (!hasFarPoint) {
      points.push(farPoint);
      segment.delete();
      segment = pointArrayToMat(points);
    }

    const area = contourArea(segment);
    if (area >= minArea) {
      concavities.push(segment);
    } else {
      segment.delete();
    }
  }

  defects.delete();
  return concavities;
}

function polygonArea(points) {
  if (!points || points.length < 3) {
    return 0;
  }
  let area = 0;
  for (let i = 0; i < points.length; i += 1) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area) * 0.5;
}

function isValidQuadrilateral(points, minArea = 1.0) {
  if (!Array.isArray(points) || points.length !== 4) {
    return false;
  }

  for (let i = 0; i < points.length; i += 1) {
    const pt = points[i];
    if (!Array.isArray(pt) || pt.length < 2) {
      return false;
    }
    const [x, y] = pt;
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      return false;
    }
    for (let j = i + 1; j < points.length; j += 1) {
      const other = points[j];
      if (!Array.isArray(other) || other.length < 2) {
        return false;
      }
      if (Math.hypot(x - other[0], y - other[1]) <= 1e-3) {
        return false;
      }
    }
  }

  const area = polygonArea(points);
  return Number.isFinite(area) && area > minArea;
}

function pointsInsideContour(points, contour) {
  if (!Array.isArray(points) || points.length === 0 || !contour || contour.rows === 0) {
    return false;
  }
  return points.every((pt) => (
    Number.isFinite(pt[0]) &&
    Number.isFinite(pt[1]) &&
    cv.pointPolygonTest(contour, new cv.Point(pt[0], pt[1]), false) >= -1e-3
  ));
}

function orientation(a, b, c) {
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const acx = c[0] - a[0];
  const acy = c[1] - a[1];
  return abx * acy - aby * acx;
}

function segmentsProperlyIntersect(p1, p2, q1, q2, eps = 1e-5) {
  const o1 = orientation(p1, p2, q1);
  const o2 = orientation(p1, p2, q2);
  const o3 = orientation(q1, q2, p1);
  const o4 = orientation(q1, q2, p2);

  if (
    ((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps)) &&
    ((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps))
  ) {
    return true;
  }
  return false;
}

function quadWithinConcavity(quad, concavity) {
  const contour = concavity.clone();
  const points = quad.map((pt) => [pt[0], pt[1]]);
  if (!pointsInsideContour(points, contour)) {
    contour.delete();
    return false;
  }

  const concavityPoints = matToPointArray(concavity);
  if (concavityPoints.length >= 2) {
    for (let i = 0; i < 4; i += 1) {
      const p1 = points[i];
      const p2 = points[(i + 1) % 4];
      for (let j = 0; j < concavityPoints.length; j += 1) {
        const q1 = concavityPoints[j];
        const q2 = concavityPoints[(j + 1) % concavityPoints.length];
        if (segmentsProperlyIntersect(p1, p2, q1, q2)) {
          contour.delete();
          return false;
        }
      }
    }
  }

  for (let i = 0; i < 4; i += 1) {
    const start = points[i];
    const end = points[(i + 1) % 4];
    for (const t of [0.25, 0.5, 0.75]) {
      const sample = [
        (1 - t) * start[0] + t * end[0],
        (1 - t) * start[1] + t * end[1]
      ];
      if (
        cv.pointPolygonTest(concavity, new cv.Point(sample[0], sample[1]), false) < -1e-3
      ) {
        contour.delete();
        return false;
      }
    }
  }

  const midpoints = [];
  for (let i = 0; i < 4; i += 1) {
    const next = points[(i + 1) % 4];
    midpoints.push([(points[i][0] + next[0]) * 0.5, (points[i][1] + next[1]) * 0.5]);
  }
  const centroid = [
    (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0,
    (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0
  ];
  const triangleCentroids = [
    [
      (points[0][0] + points[1][0] + points[2][0]) / 3.0,
      (points[0][1] + points[1][1] + points[2][1]) / 3.0
    ],
    [
      (points[0][0] + points[2][0] + points[3][0]) / 3.0,
      (points[0][1] + points[2][1] + points[3][1]) / 3.0
    ]
  ];

  const ok = pointsInsideContour(midpoints.concat([centroid]).concat(triangleCentroids), contour);
  contour.delete();
  return ok;
}

function isConvexQuad(quad) {
  if (!quad || quad.length !== 4) {
    return false;
  }
  let crossSign = 0;
  for (let i = 0; i < 4; i += 1) {
    const a = quad[i];
    const b = quad[(i + 1) % 4];
    const c = quad[(i + 2) % 4];
    const cross = orientation(a, b, c);
    if (Math.abs(cross) < 1e-6) {
      continue;
    }
    const sign = Math.sign(cross);
    if (crossSign === 0) {
      crossSign = sign;
    } else if (sign !== crossSign) {
      return false;
    }
  }
  return polygonArea(quad) > 1e-3;
}

function resampleConcavityVertices(concavity, maxVertices = 24, step = 8.0) {
  if (!concavity || concavity.rows === 0) {
    return [];
  }

  const perimeter = Math.max(cv.arcLength(concavity, true), 1.0);
  const epsilon = 0.01 * perimeter;
  const approx = new cv.Mat();
  cv.approxPolyDP(concavity, approx, epsilon, true);
  const base = approx.rows >= 4 ? approx.clone() : concavity.clone();
  approx.delete();

  const points = matToPointArray(base);
  base.delete();

  if (points.length === 0) {
    return [];
  }

  const densified = [];
  for (let i = 0; i < points.length; i += 1) {
    const p1 = points[i];
    const p2 = points[(i + 1) % points.length];
    densified.push(p1);
    const dist = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
    if (dist <= step) {
      continue;
    }
    const subdivisions = Math.floor(dist / step);
    for (let s = 1; s <= subdivisions; s += 1) {
      const t = s / (subdivisions + 1);
      densified.push([
        (1 - t) * p1[0] + t * p2[0],
        (1 - t) * p1[1] + t * p2[1]
      ]);
    }
  }

  let selected = densified;
  if (densified.length > maxVertices && maxVertices > 0) {
    selected = [];
    for (let i = 0; i < maxVertices; i += 1) {
      const idx = Math.floor((i * densified.length) / maxVertices);
      selected.push(densified[Math.min(idx, densified.length - 1)]);
    }
  }

  const unique = [];
  selected.forEach((pt) => {
    if (
      unique.length === 0 ||
      Math.hypot(pt[0] - unique[unique.length - 1][0], pt[1] - unique[unique.length - 1][1]) > 1e-3
    ) {
      unique.push(pt);
    }
  });

  return unique;
}

function largestQuadrilateralInConcavity(concavity, maxVertices = 24, samplingStep = 8.0) {
  if (!concavity || concavity.rows < 4) {
    return null;
  }

  const candidates = resampleConcavityVertices(concavity, maxVertices, samplingStep);
  if (candidates.length < 4) {
    return null;
  }

  let bestQuad = null;
  let bestArea = 0;

  for (let i = 0; i < candidates.length - 3; i += 1) {
    for (let j = i + 1; j < candidates.length - 2; j += 1) {
      for (let k = j + 1; k < candidates.length - 1; k += 1) {
        for (let l = k + 1; l < candidates.length; l += 1) {
          const quad = [
            candidates[i],
            candidates[j],
            candidates[k],
            candidates[l]
          ];
          if (!isConvexQuad(quad)) {
            continue;
          }
          if (!quadWithinConcavity(quad, concavity)) {
            continue;
          }
          const area = polygonArea(quad);
          if (area > bestArea) {
            bestArea = area;
            bestQuad = quad.map((pt) => [pt[0], pt[1]]);
          }
        }
      }
    }
  }

  return bestQuad;
}

function findPointsFromContour(contour) {
  const epsilon = 0.002 * cv.arcLength(contour, true);
  const approx = new cv.Mat();
  cv.approxPolyDP(contour, approx, epsilon, true);
  const hull = new cv.Mat();
  cv.convexHull(approx, hull, false, true);

  let working = null;
  if (hull && hull.rows > 0) {
    working = hull.clone();
  } else {
    working = approx.clone();
  }

  hull.delete();
  approx.delete();

  if (!working || working.rows < 2) {
    if (working) {
      working.delete();
    }
    return { points: null, approx: null };
  }

  const approxPoints = matToPointArray(working);
  if (approxPoints.length < 2) {
    working.delete();
    return { points: null, approx: null };
  }

  const byLength = [];
  const byHeight = [];
  const byHeightDesc = [];

  for (let i = 0; i < approxPoints.length; i += 1) {
    const pt1 = approxPoints[i];
    const pt2 = approxPoints[(i + 1) % approxPoints.length];
    const dx = pt2[0] - pt1[0];
    const dy = pt2[1] - pt1[1];
    const length = Math.hypot(dx, dy);
    const height = Math.min(pt1[1], pt2[1]);
    let angle = Math.atan2(dy, dx) * (180 / Math.PI);
    angle = (-angle) % 180;
    if (angle < 0) {
      angle += 180;
    }
    if (angle < 80 || angle > 100) {
      if (Math.abs(dx) > 20) {
        byLength.push([length, height, dx, dy, pt1, pt2]);
        byHeight.push([length, height, dx, dy, pt1, pt2]);
        byHeightDesc.push([length, height, dx, dy, pt1, pt2]);
      }
    }
  }

  if (byLength.length === 0) {
    return { points: null, approx: working };
  }

  byLength.sort((a, b) => b[0] - a[0]);
  byHeight.sort((a, b) => b[1] - a[1]);
  byHeightDesc.sort((a, b) => a[1] - b[1]);

  const highest = byHeightDesc[0];
  const lowest = byHeight[0];

  if (!highest || !lowest) {
    return { points: null, approx: working };
  }

  const points = [highest[4], highest[5], lowest[4], lowest[5]];

  console.log(`Line 1: ${JSON.stringify([highest[4], highest[5]])}`);
  console.log(`Line 2: ${JSON.stringify([lowest[4], lowest[5]])}`);

  return { points, approx: working };
}

function validateLine(line) {
  const [[x1, y1], [x2, y2]] = line;
  if (Math.hypot(x2 - x1, y2 - y1) <= 1e-9) {
    throw new Error('Chaque ligne doit être définie par deux points distincts.');
  }
  const angle = normalizedAngleDeg(line);
  if (angle >= 75.0 - 1e-9 && angle <= 105.0 + 1e-9) {
    throw new Error('Les segments compris entre 75° et 105° sont considérés comme verticaux.');
  }
}

function segmentLength(line) {
  const [[x1, y1], [x2, y2]] = line;
  return Math.hypot(x2 - x1, y2 - y1);
}

function normalizedAngleDeg(line) {
  const [[x1, y1], [x2, y2]] = line;
  const angle = Math.atan2(y2 - y1, x2 - x1) * (180 / Math.PI);
  return (angle + 180) % 180;
}

function extendLine(line, factor) {
  if (factor <= 0) {
    throw new Error("Le facteur d'extension doit être strictement positif.");
  }
  const [[x1, y1], [x2, y2]] = line;
  const length = segmentLength(line);
  const midX = (x1 + x2) / 2.0;
  const midY = (y1 + y2) / 2.0;
  const unitDx = (x2 - x1) / length;
  const unitDy = (y2 - y1) / length;
  const half = (length * factor) / 2.0;
  return [
    [midX - unitDx * half, midY - unitDy * half],
    [midX + unitDx * half, midY + unitDy * half]
  ];
}

function lineParameters(line) {
  const [[x1, y1], [x2, y2]] = line;
  const slope = (y2 - y1) / (x2 - x1);
  const intercept = y1 - slope * x1;
  return [slope, intercept];
}

function asPointArray(points) {
  if (!points) {
    return [];
  }
  return points.map((pt) => [pt[0], pt[1]]);
}

function orderPoints(points) {
  if (points.length !== 4) {
    throw new Error("Quatre points sont requis pour l'ordonnancement.");
  }
  const sum = points.map((pt) => pt[0] + pt[1]);
  const diff = points.map((pt) => pt[0] - pt[1]);
  const ordered = new Array(4);
  ordered[0] = points[sum.indexOf(Math.min(...sum))];
  ordered[2] = points[sum.indexOf(Math.max(...sum))];
  ordered[1] = points[diff.indexOf(Math.min(...diff))];
  ordered[3] = points[diff.indexOf(Math.max(...diff))];
  return ordered;
}

function minimumBoundingQuadrilateral(points) {
  const pts = asPointArray(points);
  if (pts.length === 0) {
    throw new Error('Le contour doit contenir au moins un point.');
  }
  if (pts.length === 1) {
    return [pts[0], pts[0], pts[0], pts[0]];
  }
  if (pts.length === 2) {
    const [p1, p2] = pts;
    const minX = Math.min(p1[0], p2[0]);
    const maxX = Math.max(p1[0], p2[0]);
    const minY = Math.min(p1[1], p2[1]);
    const maxY = Math.max(p1[1], p2[1]);
    return [
      [minX, minY],
      [minX, maxY],
      [maxX, maxY],
      [maxX, minY]
    ];
  }

  const contourMat = pointArrayToMat(pts, cv.CV_32FC2);
  const hull = new cv.Mat();
  cv.convexHull(contourMat, hull, false, true);
  contourMat.delete();
  const hullPoints = matToPointArray(hull);
  hull.delete();

  if (hullPoints.length <= 2) {
    return minimumBoundingQuadrilateral(hullPoints);
  }

  let bestArea = Number.POSITIVE_INFINITY;
  let bestCorners = null;

  for (let i = 0; i < hullPoints.length; i += 1) {
    const p1 = hullPoints[i];
    const p2 = hullPoints[(i + 1) % hullPoints.length];
    const angle = Math.atan2(p2[1] - p1[1], p2[0] - p1[0]);
    const cos = Math.cos(-angle);
    const sin = Math.sin(-angle);
    const rotated = hullPoints.map((pt) => [
      pt[0] * cos - pt[1] * sin,
      pt[0] * sin + pt[1] * cos
    ]);
    const xs = rotated.map((pt) => pt[0]);
    const ys = rotated.map((pt) => pt[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const area = (maxX - minX) * (maxY - minY);
    if (area < bestArea - 1e-6) {
      bestArea = area;
      const rect = [
        [minX, minY],
        [minX, maxY],
        [maxX, maxY],
        [maxX, minY]
      ];
      const cosInv = Math.cos(angle);
      const sinInv = Math.sin(angle);
      bestCorners = rect.map((pt) => [
        pt[0] * cosInv - pt[1] * sinInv,
        pt[0] * sinInv + pt[1] * cosInv
      ]);
    }
  }

  if (!bestCorners) {
    throw new Error('Échec du calcul du quadrilatère minimal.');
  }

  return orderPoints(bestCorners);
}

function quadrilateralFromLines(line1, line2, extensionFactor = 100.0, contour = null) {
  if (contour) {
    return minimumBoundingQuadrilateral(contour);
  }
  if (!line1 || !line2) {
    throw new Error("line1 et line2 doivent être fournis si aucun contour n'est donné.");
  }

  validateLine(line1);
  validateLine(line2);

  const length1 = segmentLength(line1);
  const length2 = segmentLength(line2);
  const longest = length1 >= length2 ? line1 : line2;
  const minX = Math.min(longest[0][0], longest[1][0]);
  const maxX = Math.max(longest[0][0], longest[1][0]);

  const extended1 = extendLine(line1, extensionFactor);
  const extended2 = extendLine(line2, extensionFactor);

  const [slope1, intercept1] = lineParameters(extended1);
  const [slope2, intercept2] = lineParameters(extended2);

  const leftPoints = [
    [minX, slope1 * minX + intercept1],
    [minX, slope2 * minX + intercept2]
  ];
  const rightPoints = [
    [maxX, slope1 * maxX + intercept1],
    [maxX, slope2 * maxX + intercept2]
  ];

  leftPoints.sort((a, b) => a[1] - b[1]);
  rightPoints.sort((a, b) => a[1] - b[1]);

  const [topLeft, bottomLeft] = leftPoints;
  const [topRight, bottomRight] = rightPoints;

  return [topLeft, bottomLeft, bottomRight, topRight];
}

function polygonInsideMask(polygon, mask) {
  if (!mask || mask.rows === 0 || mask.cols === 0) {
    return false;
  }
  const polyMat = pointArrayToMat(polygon);
  const fill = cv.Mat.zeros(mask.rows, mask.cols, cv.CV_8UC1);
  const vec = new cv.MatVector();
  vec.push_back(polyMat);
  cv.fillPoly(fill, vec, new cv.Scalar(255));
  vec.delete();
  polyMat.delete();

  const overlap = new cv.Mat();
  cv.bitwise_and(fill, mask, overlap);
  const diff = new cv.Mat();
  cv.subtract(fill, overlap, diff);
  const inside = cv.countNonZero(diff) === 0;
  fill.delete();
  overlap.delete();
  diff.delete();
  return inside;
}

function largestInscribedQuadrilateral(diffContour, diffMask) {
  if (!diffContour || !diffMask || diffMask.rows === 0 || diffMask.cols === 0) {
    return null;
  }
  const points = matToPointArray(diffContour);
  if (points.length < 4) {
    return null;
  }

  const perimeter = cv.arcLength(diffContour, true);
  const epsilon = Math.max(1.0, 0.01 * perimeter);
  const approx = new cv.Mat();
  cv.approxPolyDP(diffContour, approx, epsilon, true);
  let candidatePoints = matToPointArray(approx);
  approx.delete();

  if (candidatePoints.length < 4) {
    candidatePoints = points;
  }

  const candidateMat = pointArrayToMat(candidatePoints, cv.CV_32FC2);
  const hull = new cv.Mat();
  cv.convexHull(candidateMat, hull, false, true);
  candidateMat.delete();
  const hullPoints = matToPointArray(hull);
  hull.delete();

  if (hullPoints.length === 4 && polygonInsideMask(hullPoints, diffMask)) {
    return hullPoints;
  }

  if (hullPoints.length < 4) {
    return null;
  }

  if (candidatePoints.length > 24) {
    const sampled = [];
    for (let i = 0; i < 24; i += 1) {
      const idx = Math.floor((i * candidatePoints.length) / 24);
      sampled.push(candidatePoints[Math.min(idx, candidatePoints.length - 1)]);
    }
    candidatePoints = sampled;
  }

  let bestQuad = null;
  let bestArea = 0;

  for (let i = 0; i < candidatePoints.length - 3; i += 1) {
    for (let j = i + 1; j < candidatePoints.length - 2; j += 1) {
      for (let k = j + 1; k < candidatePoints.length - 1; k += 1) {
        for (let l = k + 1; l < candidatePoints.length; l += 1) {
          const quad = [
            candidatePoints[i],
            candidatePoints[j],
            candidatePoints[k],
            candidatePoints[l]
          ];
          const quadMat = pointArrayToMat(quad, cv.CV_32FC2);
          const hullQuad = new cv.Mat();
          cv.convexHull(quadMat, hullQuad, false, true);
          quadMat.delete();
          const hullQuadPoints = matToPointArray(hullQuad);
          hullQuad.delete();
          if (hullQuadPoints.length !== 4) {
            continue;
          }
          if (!polygonInsideMask(hullQuadPoints, diffMask)) {
            continue;
          }
          const area = polygonArea(hullQuadPoints);
          if (area > bestArea) {
            bestArea = area;
            bestQuad = hullQuadPoints;
          }
        }
      }
    }
  }

  return bestQuad;
}

function detectLargestHollowParallelepiped(contour, imageShape = null, minArea = 0.0) {
  if (!contour || contour.rows < 3) {
    return null;
  }

  const epsilon = 0.002 * cv.arcLength(contour, true);
  const approx = new cv.Mat();
  cv.approxPolyDP(contour, approx, epsilon, true);
  const hull = new cv.Mat();
  cv.convexHull(contour, hull, false, true);

  const contourPoints = matToPointArray(contour);
  const hullPoints = matToPointArray(hull);

  let maskRows;
  let maskCols;
  let offset = [0, 0];
  let contourMaskPoints;
  let hullMaskPoints;

  if (!imageShape) {
    const hullMat = pointArrayToMat(hullPoints);
    const rect = cv.boundingRect(hullMat);
    hullMat.delete();
    const padding = 2;
    offset = [rect.x - padding, rect.y - padding];
    maskRows = rect.height + 2 * padding;
    maskCols = rect.width + 2 * padding;
    contourMaskPoints = contourPoints.map((pt) => [pt[0] - offset[0], pt[1] - offset[1]]);
    hullMaskPoints = hullPoints.map((pt) => [pt[0] - offset[0], pt[1] - offset[1]]);
  } else {
    maskRows = imageShape.rows;
    maskCols = imageShape.cols;
    contourMaskPoints = contourPoints;
    hullMaskPoints = hullPoints;
  }

  if (maskRows <= 0 || maskCols <= 0) {
    approx.delete();
    hull.delete();
    return null;
  }

  const hullMask = cv.Mat.zeros(maskRows, maskCols, cv.CV_8UC1);
  const hullVec = new cv.MatVector();
  const hullMaskMat = pointArrayToMat(hullMaskPoints);
  hullVec.push_back(hullMaskMat);
  cv.drawContours(hullMask, hullVec, 0, new cv.Scalar(255, 255, 255, 255), cv.FILLED);
  hullVec.delete();
  hullMaskMat.delete();

  const contourMask = cv.Mat.zeros(maskRows, maskCols, cv.CV_8UC1);
  const contourVec = new cv.MatVector();
  const contourMaskMat = pointArrayToMat(contourMaskPoints);
  contourVec.push_back(contourMaskMat);
  cv.drawContours(contourMask, contourVec, 0, new cv.Scalar(255, 255, 255, 255), cv.FILLED);
  contourVec.delete();
  contourMaskMat.delete();

  const contourMaskInv = new cv.Mat();
  cv.bitwise_not(contourMask, contourMaskInv);
  const diffMask = new cv.Mat();
  cv.bitwise_and(hullMask, contourMaskInv, diffMask);
  contourMask.delete();
  contourMaskInv.delete();
  hullMask.delete();

  const diffContours = new cv.MatVector();
  const diffHierarchy = new cv.Mat();
  const diffMaskForContours = diffMask.clone();
  cv.findContours(diffMaskForContours, diffContours, diffHierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  diffMaskForContours.delete();
  diffHierarchy.delete();

  let bestCandidate = null;
  let bestArea = minArea;

  for (let i = 0; i < diffContours.size(); i += 1) {
    const diffCnt = diffContours.get(i);
    const quadLocal = largestInscribedQuadrilateral(diffCnt, diffMask);
    if (!quadLocal) {
      diffCnt.delete();
      continue;
    }

    const area = polygonArea(quadLocal);
    if (area < minArea) {
      diffCnt.delete();
      continue;
    }

    const quadGlobal = quadLocal.map((pt) => [pt[0] + offset[0], pt[1] + offset[1]]);
    const diffCntGlobal = translateContour(diffCnt, offset[0], offset[1]);

    const quadMatLocal = pointArrayToMat(quadLocal, cv.CV_32FC2);
    const rectInfo = cv.minAreaRect(quadMatLocal);
    quadMatLocal.delete();

    const candidate = {
      box: quadGlobal,
      area,
      rect: {
        center: [rectInfo.center.x + offset[0], rectInfo.center.y + offset[1]],
        size: [rectInfo.size.width, rectInfo.size.height],
        angle: rectInfo.angle
      },
      diffContour: diffCntGlobal
    };

    if (area > bestArea) {
      if (bestCandidate && bestCandidate.diffContour) {
        bestCandidate.diffContour.delete();
      }
      bestCandidate = candidate;
      bestArea = area;
    } else {
      diffCntGlobal.delete();
    }

    diffCnt.delete();
  }

  diffContours.delete();
  approx.delete();
  hull.delete();
  diffMask.delete();

  if (!bestCandidate) {
    return null;
  }

  return { best: bestCandidate };
}

function projectTexture(background, texture, dstPoints, clipMask = null) {
  if (!dstPoints || dstPoints.length !== 4) {
    throw new Error('dstPoints must contain exactly four points');
  }

  const srcPoints = [
    [0, 0],
    [texture.cols - 1, 0],
    [texture.cols - 1, texture.rows - 1],
    [0, texture.rows - 1]
  ];

  const srcMat = pointArrayToMat(srcPoints, cv.CV_32FC2);
  const dstMat = pointArrayToMat(dstPoints, cv.CV_32FC2);
  const homography = cv.getPerspectiveTransform(srcMat, dstMat);
  srcMat.delete();
  dstMat.delete();

  const warped = new cv.Mat();
  cv.warpPerspective(texture, warped, homography, new cv.Size(background.cols, background.rows));

  const mask = new cv.Mat(texture.rows, texture.cols, cv.CV_8UC1, new cv.Scalar(255));
  const warpedMask = new cv.Mat();
  cv.warpPerspective(mask, warpedMask, homography, new cv.Size(background.cols, background.rows));
  mask.delete();
  homography.delete();

  if (clipMask) {
    const clip = clipMask.clone();
    if (clip.type() !== cv.CV_8UC1) {
      clip.convertTo(clip, cv.CV_8UC1);
    }
    cv.bitwise_and(warpedMask, clip, warpedMask);
    cv.bitwise_and(warped, warped, warped, warpedMask);
    clip.delete();
  }

  const invMask = new cv.Mat();
  cv.bitwise_not(warpedMask, invMask);

  const backgroundMasked = new cv.Mat();
  cv.bitwise_and(background, background, backgroundMasked, invMask);

  const output = new cv.Mat();
  cv.add(backgroundMasked, warped, output);

  warped.delete();
  warpedMask.delete();
  invMask.delete();
  backgroundMasked.delete();

  return output;
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