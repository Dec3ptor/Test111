let isDomReady = false;

// Initialize OpenCV.js
function initialize() {
  if (typeof cv === "undefined") {
    console.log("Waiting for OpenCV.js to load...");
    setTimeout(initialize, 50);
    return;
  }
  console.log("OpenCV.js is loaded.");
  document.getElementById("status").innerHTML = "OpenCV.js is ready.";
  if (isDomReady) {
    main();
  }
}

// Run the main function after the DOM is initialized
document.addEventListener("DOMContentLoaded", function () {
  isDomReady = true;
  if (typeof cv !== "undefined") {
    main();
  } else {
    document.getElementById("status").innerHTML = "OpenCV.js is not ready.";
  }
});

function loadAndProcessImage() {
  let inputElement = document.getElementById("fileInput");
  let file = inputElement.files[0];
  if (file) {
    let reader = new FileReader();
    reader.onload = function (event) {
      let imageElement = document.getElementById("imageSrc1");
      if (!imageElement) {
        imageElement = document.createElement("img");
        imageElement.id = "imageSrc1";
        document.body.appendChild(imageElement);
      }
      imageElement.src = event.target.result;
      imageElement.onload = function () {
        main();
      };
    };
    reader.readAsDataURL(file);
  } else {
    console.log("No file selected.");
  }
}

//----------------------------------//

// Load the image and split channels
function loadImageAndSplitChannels(imageId) {
  let src = cv.imread(imageId);
  let rgbaPlanes = new cv.MatVector();
  cv.split(src, rgbaPlanes);
  return rgbaPlanes;
}

// Calculate mean for each channel
function calculateMean(rgbaPlanes) {
  let r = rgbaPlanes.get(0);
  let g = rgbaPlanes.get(1);
  let b = rgbaPlanes.get(2);
  let rMean = cv.mean(r)[0];
  let gMean = cv.mean(g)[0];
  let bMean = cv.mean(b)[0];
  return { rMean, gMean, bMean };
}

// Perform color compensation
function colorCompensation(rgbaPlanes, means) {
  let r = rgbaPlanes.get(0);
  let g = rgbaPlanes.get(1);
  let b = rgbaPlanes.get(2);
  let alpha = 0.1;
  cv.addWeighted(r, 1, g, alpha, means.gMean - means.rMean, r);
  alpha = 0;
  cv.addWeighted(b, 1, g, alpha, means.gMean - means.bMean, b);
}

// Merge back to color image
function mergeToColorImage(rgbaPlanes, dst) {
  cv.merge(rgbaPlanes, dst);
}

// White Balance using Gray World Assumption
function whiteBalance(src) {
  let dst = new cv.Mat();
  let rgbaPlanes = new cv.MatVector();
  cv.split(src, rgbaPlanes);

  let r = rgbaPlanes.get(0);
  let g = rgbaPlanes.get(1);
  let b = rgbaPlanes.get(2);

  // Calculate the mean of each channel
  let r_mean = cv.mean(r)[0];
  let g_mean = cv.mean(g)[0];
  let b_mean = cv.mean(b)[0];

  // Calculate the scaling factors
  let mean = (r_mean + g_mean + b_mean) / 3;
  let r_scale = mean / r_mean;
  let g_scale = mean / g_mean;
  let b_scale = mean / b_mean;

  // Scale each channel
  cv.convertScaleAbs(r, r, r_scale);
  cv.convertScaleAbs(g, g, g_scale);
  cv.convertScaleAbs(b, b, b_scale);

  // Create a new MatVector and populate it with the modified channels
  let newRgbaPlanes = new cv.MatVector();
  newRgbaPlanes.push_back(r);
  newRgbaPlanes.push_back(g);
  newRgbaPlanes.push_back(b);

  // Merge the channels back
  cv.merge(newRgbaPlanes, dst);

  // Clean up
  rgbaPlanes.delete();
  newRgbaPlanes.delete();
  r.delete();
  g.delete();
  b.delete();

  return dst;
}

// Gamma Correction
function gammaCorrection(src, gamma) {
  let dst = new cv.Mat();
  src.convertTo(dst, cv.CV_32F, 1 / 255.0); // Convert to float and normalize to [0, 1]

  // Apply gamma correction
  cv.pow(dst, gamma, dst);

  // Convert back to original type and range
  dst.convertTo(dst, src.type(), 255.0);

  return dst;
}

// Image Sharpening
function imageSharpening(src, sigma, N) {
  let dst = new cv.Mat();
  let temp = src.clone();
  let gauss = new cv.Mat();
  for (let iter = 1; iter <= N; iter++) {
    cv.GaussianBlur(
      temp,
      gauss,
      new cv.Size(0, 0),
      sigma,
      sigma,
      cv.BORDER_DEFAULT
    );
    cv.min(src, gauss, temp);
  }
  let gain = 1;
  cv.subtract(src, temp, dst);
  cv.convertScaleAbs(dst, dst, gain);
  cv.cvtColor(dst, dst, cv.COLOR_BGR2GRAY);
  cv.equalizeHist(dst, dst);
  return dst;
}

// Weights Calculation
function weightsCalculation(src1, src2) {
  let WC1 = new cv.Mat();
  let WC2 = new cv.Mat();
  cv.Laplacian(src1, WC1, cv.CV_64F);
  cv.Laplacian(src2, WC2, cv.CV_64F);
  // Assuming saliency weight and saturation weight are both 1 for simplicity
  let WS1 = 1;
  let WS2 = 1;
  let WSAT1 = 1;
  let WSAT2 = 1;
  let W1 =
    (WC1 + WS1 + WSAT1 + 0.1) / (WC1 + WS1 + WSAT1 + WC2 + WS2 + WSAT2 + 0.2);
  let W2 =
    (WC2 + WS2 + WSAT2 + 0.1) / (WC1 + WS1 + WSAT1 + WC2 + WS2 + WSAT2 + 0.2);
  return { W1, W2 };
}

// Multi-Scale Fusion
function multiScaleFusion(img1, img2) {
  let level = 10;
  let Weight1 = gaussianPyramid(img1, level);
  let Weight2 = gaussianPyramid(img2, level);
  let LapPyr1 = laplacianPyramid(img1, level);
  let LapPyr2 = laplacianPyramid(img2, level);
  let fusedPyr = [];
  for (let k = 0; k < level; k++) {
    let Rr = cv.addWeighted(LapPyr1[k], Weight1[k], LapPyr2[k], Weight2[k], 0);
    fusedPyr.push(Rr);
  }
  let fusedImg = pyramidReconstruct(fusedPyr);
  return fusedImg;
}

// Helper function to calculate Gaussian Pyramid
function gaussianPyramid(img, level) {
  let pyramid = [];
  let temp = img.clone();
  for (let i = 0; i < level; i++) {
    let down = new cv.Mat();
    cv.pyrDown(temp, down);
    pyramid.push(down);
    temp = down;
  }
  return pyramid;
}

// Helper function to calculate Laplacian Pyramid
function laplacianPyramid(img, level) {
  let pyramid = [];
  let temp = img.clone();
  for (let i = 0; i < level; i++) {
    let down = new cv.Mat();
    let up = new cv.Mat();
    cv.pyrDown(temp, down);
    cv.pyrUp(down, up, temp.size());
    let laplacian = new cv.Mat();
    cv.subtract(temp, up, laplacian);
    pyramid.push(laplacian);
    temp = down;
  }
  return pyramid;
}

// Helper function to reconstruct image from pyramid
function pyramidReconstruct(pyramid) {
  let reconstructed = pyramid[pyramid.length - 1];
  for (let i = pyramid.length - 2; i >= 0; i--) {
    let up = new cv.Mat();
    cv.pyrUp(reconstructed, up, pyramid[i].size());
    cv.add(up, pyramid[i], reconstructed);
  }
  return reconstructed;
}

//----------------------------------//

function main() {
  console.log("Main function is called.");

  if (typeof cv === "undefined") {
    console.log("OpenCV.js is not loaded yet.");
    return;
  }

  let imgElement = document.getElementById("imageSrc1"); // New line
  if (!imgElement || !document.getElementById("canvasOutput")) {
    console.log("No images loaded.");
    document.getElementById("status").innerHTML = "No images loaded.";
    return;
  }

  // New lines: Check if the image is loaded and has non-zero dimensions
  if (imgElement.naturalWidth === 0 || imgElement.naturalHeight === 0) {
    console.log("Image has not loaded or has zero dimensions.");
    return;
  }

  try {
    // Load and process the image
    let src1 = cv.imread("imageSrc1");
    cv.imshow("canvasOutput1", src1); // Display original image

    let rgbaPlanes1 = loadImageAndSplitChannels("imageSrc1");
    let means1 = calculateMean(rgbaPlanes1);
    colorCompensation(rgbaPlanes1, means1);

    let wbImage1 = whiteBalance(src1);
    cv.imshow("canvasOutput2", wbImage1); // Display white-balanced image

    let gamma1 = 5;
    let gammaCorrectedImage1 = gammaCorrection(wbImage1, gamma1);
    cv.imshow("canvasOutput3", gammaCorrectedImage1); // Display gamma-corrected image

    let sigma1 = 20;
    let N1 = 30;
    let sharpenedImage1 = imageSharpening(gammaCorrectedImage1, sigma1, N1);
    cv.imshow("canvasOutput4", sharpenedImage1); // Display sharpened image

    // Clean up
    src1.delete();
    wbImage1.delete();
    gammaCorrectedImage1.delete();
    sharpenedImage1.delete();

    console.log("Result displayed and resources cleaned up.");
  } catch (e) {
    console.error("An error occurred: ", e.message, e.stack);
  }
}

// Call initialize to make sure OpenCV.js is loaded
initialize();
