// ============================================
// DOCTOR'S HANDWRITTEN PRESCRIPTION AI
// FRONTEND JAVASCRIPT
// ============================================


// ============================================
// GET HTML ELEMENTS
// ============================================

const imageInput = document.getElementById("prescriptionImage");

const preview = document.getElementById("preview");

const analyzeBtn = document.getElementById("analyzeBtn");

const resultText = document.getElementById("resultText");

const cameraBtn = document.getElementById("cameraBtn");

const cameraContainer =
    document.getElementById("camera-container");

const camera =
    document.getElementById("camera");

const captureBtn =
    document.getElementById("captureBtn");

const closeCameraBtn =
    document.getElementById("closeCameraBtn");


// ============================================
// VARIABLES
// ============================================

// Stores the uploaded/captured image
let selectedImage = null;

// Stores camera stream
let cameraStream = null;


// ============================================
// 1. UPLOAD PRESCRIPTION
// ============================================

imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (!file) {
        return;
    }

    // Check whether selected file is an image
    if (!file.type.startsWith("image/")) {

        resultText.textContent =
            "Please select an image file.";

        return;
    }

    // Store image
    selectedImage = file;

    // Create preview URL
    const imageURL =
        URL.createObjectURL(file);

    // Display image
    preview.src = imageURL;

    preview.style.display = "block";

    // Display message
    resultText.textContent =
        "Prescription uploaded successfully. You can now analyze it.";

});


// ============================================
// 2. OPEN CAMERA
// ============================================

cameraBtn.addEventListener("click", async function () {

    try {

        // Check browser camera support
        if (!navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia) {

            resultText.textContent =
                "Camera is not supported by this browser.";

            return;
        }


        // Ask permission and open camera
        cameraStream =
            await navigator.mediaDevices.getUserMedia({

                video: {
                    facingMode: {
                        ideal: "environment"
                    }
                },

                audio: false

            });


        // Connect camera stream to video element
        camera.srcObject =
            cameraStream;


        // Show camera section
        cameraContainer.style.display =
            "block";


        // Message
        resultText.textContent =
            "Camera opened. Place the prescription clearly inside the camera view.";

    }

    catch (error) {

        console.error(
            "Camera error:",
            error
        );


        resultText.textContent =
            "Camera access was denied or unavailable. Please allow camera permission and try again.";

    }

});


// ============================================
// 3. CAPTURE PHOTO
// ============================================

captureBtn.addEventListener("click", function () {

    // Make sure camera is running
    if (!cameraStream) {

        resultText.textContent =
            "Camera is not active.";

        return;
    }


    // Create canvas
    const canvas =
        document.createElement("canvas");


    // Use camera resolution
    canvas.width =
        camera.videoWidth;

    canvas.height =
        camera.videoHeight;


    // Get drawing context
    const context =
        canvas.getContext("2d");


    // Draw current camera frame
    context.drawImage(
        camera,
        0,
        0,
        canvas.width,
        canvas.height
    );


    // Convert image to JPEG
    canvas.toBlob(function (blob) {

        if (!blob) {

            resultText.textContent =
                "Unable to capture the photo.";

            return;
        }


        // Create File object
        selectedImage =
            new File(
                [blob],
                "captured-prescription.jpg",
                {
                    type: "image/jpeg"
                }
            );


        // Create preview URL
        const imageURL =
            URL.createObjectURL(blob);


        // Display captured image
        preview.src =
            imageURL;

        preview.style.display =
            "block";


        // Display success message
        resultText.textContent =
            "Photo captured successfully. You can now analyze the prescription.";


        // Stop camera
        stopCamera();

    }, "image/jpeg", 0.9);

});


// ============================================
// 4. CLOSE CAMERA
// ============================================

closeCameraBtn.addEventListener(
    "click",
    function () {

        stopCamera();

        resultText.textContent =
            "Camera closed.";

    }
);


// ============================================
// 5. STOP CAMERA FUNCTION
// ============================================

function stopCamera() {

    if (cameraStream) {

        // Stop every camera track
        cameraStream
            .getTracks()
            .forEach(function (track) {

                track.stop();

            });


        // Remove stream
        cameraStream = null;
    }


    // Remove video source
    camera.srcObject = null;


    // Hide camera section
    cameraContainer.style.display =
        "none";
}


// ============================================
// 6. ANALYZE PRESCRIPTION
// ============================================

analyzeBtn.addEventListener(
    "click",
    async function () {

        // Check whether image exists
        if (!selectedImage) {

            resultText.textContent =
                "Please upload or capture a prescription first.";

            return;
        }


        // Show loading state
        analyzeBtn.disabled = true;

        analyzeBtn.textContent =
            "⏳ Analyzing...";

        resultText.textContent =
            "Analyzing prescription. Please wait...";


        try {

            // Create multipart form data
            const formData =
                new FormData();

            // Backend expects the field name "file"
            formData.append(
                "file",
                selectedImage
            );


            // Send image to FastAPI backend
            const response =
                await fetch(
                    "http://127.0.0.1:8000/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            // Parse backend response
            const data =
                await response.json();


            // Handle backend errors
            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Prescription analysis failed."
                );

            }


            // Display prediction
            resultText.innerHTML = `
                <strong>Medicine:</strong>
                ${data.medicine}
                <br><br>

                <strong>Generic Name:</strong>
                ${data.generic_name}
                <br><br>

                <strong>Confidence:</strong>
                ${(data.confidence * 100).toFixed(2)}%
            `;

        }

        catch (error) {

            console.error(
                "Prediction error:",
                error
            );


            resultText.textContent =
                error.message ||
                "Unable to analyze the prescription. Please try again.";

        }

        finally {

            // Restore analyze button
            analyzeBtn.disabled = false;

            analyzeBtn.textContent =
                "🔍 Analyze Prescription";

        }

    }
);


// ============================================
// 7. STOP CAMERA WHEN PAGE CLOSES
// ============================================

window.addEventListener(
    "beforeunload",
    function () {

        stopCamera();

    }
);