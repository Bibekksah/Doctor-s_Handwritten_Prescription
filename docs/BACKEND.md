Doctor's Handwritten Prescription — Backend
1. Overview

The backend of Doctor's Handwritten Prescription provides an API for recognizing medicine names from doctors' handwritten prescription images.

The backend is responsible for:

Receiving prescription/handwritten medicine images
Validating uploaded images
Running the ML003 medicine-recognition model
Returning the predicted medicine name and confidence
Looking up the predicted medicine in the medicine database
Persisting prediction-related metadata
Providing health/status information
Providing medicine verification through the API
Handling inference and application errors safely

The backend is intended as a medicine-recognition and verification assistance system. It does not replace a doctor, pharmacist, or other qualified medical professional.

2. Technology Stack
Component	Technology
Programming language	Python 3.12.7
API framework	FastAPI
ASGI server	Uvicorn
Validation/schema	Pydantic
Database ORM	SQLAlchemy
Database migrations	Alembic
Development database	SQLite
Image processing	Pillow
Numerical processing	NumPy
ML inference	ONNX Runtime
ML model	ResNet18 / ML003
Testing	Pytest
HTTP testing	HTTPX
Version control	Git / GitHub
3. Backend Architecture

The current backend follows this workflow:

                    Frontend
                       |
                       | HTTP request
                       v
                FastAPI Application
                       |
             +---------+---------+
             |                   |
             v                   v
        Input Validation     API Routes
                                 |
                                 v
                       Prediction Service
                                 |
                         +-------+-------+
                         |               |
                         v               v
                    ML003 Inference   Database
                         |
                         v
                   ONNX Runtime
                         |
                         v
                  ResNet18 ML Model
                         |
                         v
                  Medicine Prediction
                         |
                         v
                  Medicine DB Lookup
                         |
                         v
                    JSON Response
4. Main Backend Workflow

The /predict workflow is:

1. Client uploads image
2. FastAPI receives multipart/form-data
3. Uploaded file is validated
4. Image is temporarily stored
5. PredictionService processes the image
6. ML003 ONNX model performs inference
7. Predicted class is converted to medicine label
8. Medicine database is queried
9. Prediction information is persisted
10. Temporary image file is removed
11. JSON response is returned
5. Project Structure

The relevant backend structure is:

Doctor-s_Handwritten_Prescription/
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── routes.py
│   ├── schemas.py
│   ├── utils.py
│   ├── inference.py
│   ├── ml003_inference.py
│   │
│   ├── database/
│   │   ├── database.py
│   │   └── models.py
│   │
│   ├── services/
│   │   └── prediction_service.py
│   │
│   └── exception_handlers.py
│
├── CV/
│   └── preprocessing.py
│
├── ml/
│   ├── checkpoints/
│   └── results/
│       └── optimization/
│           └── ML003_resnet18.onnx
│
├── data/
│   ├── app.db
│   └── metadata/
│       └── dataset.csv
│
├── tests/
│   └── backend/
│
├── alembic/
│
├── docs/
│
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── alembic.ini
└── README.md
6. FastAPI Application

The main FastAPI application is defined in:

backend/main.py

The application uses FastAPI lifespan management to initialize the ML003 inference engine when the application starts.

The inference engine is stored on:

app.state.inference_engine

This allows the model to be initialized once and reused across requests instead of loading the ONNX model for every prediction.

7. Application Startup

The normal application startup command is:

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

For local development, the backend can then be accessed at:

http://127.0.0.1:8000

FastAPI's interactive API documentation is available at:

http://127.0.0.1:8000/docs

The backend should be started from the project root.

8. API Endpoints

The current backend exposes the following main endpoints:

Method	Endpoint	Purpose
GET	/health	Backend/database/model health
POST	/predict	Predict medicine from uploaded image
POST	/verify	Verify medicine information
9. GET /health
Purpose

Checks whether the database and ML inference engine are available.

Request
GET /health
Example
curl http://127.0.0.1:8000/health
Successful response
{
  "status": "ok",
  "database": "ok",
  "model": "ok"
}
Degraded response

If either the database or model is unavailable:

{
  "status": "degraded",
  "database": "ok",
  "model": "error"
}

The health endpoint currently verifies:

Database connectivity
Successful initialization of the ML003 inference engine
Presence of valid model metadata
10. POST /predict
Purpose

Accepts a handwritten medicine image and returns the model's predicted medicine.

Request

The endpoint expects a multipart/form-data upload with the field:

file

Supported image types:

image/jpeg
image/png

Maximum upload size:

10 MB
Example
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@./data/metadata/Doctor’s Handwritten Prescription BD dataset/Testing/testing_words/0.png;type=image/png"
Example successful response
{
  "medicine": "Aceta",
  "generic_name": "Paracetamol",
  "confidence": 0.9699714779853821
}
Response fields
Field	Type	Description
medicine	string	Predicted medicine/class label
generic_name	string/null	Generic medicine name from database
confidence	float	Model prediction confidence between 0 and 1

The confidence value represents the model's prediction confidence. It should not be interpreted as medical certainty.

11. /predict Image Validation

Uploaded images are validated before inference.

Current validation includes:

File is not empty
File size does not exceed 10 MB
MIME type is JPEG or PNG
File contains a valid image
Image dimensions are at least 32 × 32 pixels
Image dimensions do not exceed 10000 × 10000 pixels

Invalid images return an HTTP 400 response.

Example:

{
  "detail": "Invalid image ..."
}
12. Temporary File Handling

Uploaded images are written to a temporary file before ML inference.

After processing, the temporary file is removed in a finally block.

This ensures that uploaded images are not unnecessarily retained as temporary filesystem files after the prediction request completes.

13. POST /verify
Purpose

Checks whether a medicine exists in the medicine database and returns its generic name if available.

Request
POST /verify

Content type:

application/json

Example:

{
  "medicine_name": "Aceta"
}
Example response
{
  "medicine": "Aceta",
  "exists": true,
  "generic_name": "Paracetamol"
}

If the medicine does not exist:

{
  "medicine": "UnknownMedicine",
  "exists": false,
  "generic_name": null
}
14. API Schemas

The main Pydantic schemas are defined in:

backend/schemas.py
PredictResponse
class PredictResponse(BaseModel):
    medicine: str
    generic_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
VerifyRequest
class VerifyRequest(BaseModel):
    model_config = {"str_strip_whitespace": True}
    medicine_name: str = Field(min_length=1)
VerifyResponse
class VerifyResponse(BaseModel):
    medicine: str
    exists: bool
    generic_name: str | None = None
HealthResponse
class HealthResponse(BaseModel):
    status: str
    database: str
    model: str
15. ML003 Model

The production inference model currently integrated into the backend is:

ML003

Model architecture:

ResNet18

Inference format:

ONNX

Runtime:

ONNX Runtime

Model file:

ml/results/optimization/ML003_resnet18.onnx

Model version:

ML003-resnet18-onnx-1.0.0
16. ML003 Input

The model expects:

Shape:
[batch_size, 3, 64, 256]

Data type:
float32

The model has:

78 classes
17. ML003 Preprocessing

The verified preprocessing pipeline is:

Input image
    ↓
Convert to RGB
    ↓
Grayscale conversion
    ↓
Resize with padding
    ↓
64 × 256
    ↓
Tensor conversion
    ↓
Duplicate grayscale channel to 3 channels
    ↓
Batch dimension
    ↓
ML003 ONNX model

The preprocessing used by ML003 must remain unchanged unless the model is deliberately retrained and revalidated.

18. ML003 Class Mapping

The inference system currently reconstructs the class mapping from:

data/metadata/dataset.csv

The verified class count is:

78

The checkpoint classes and CSV-derived classes have been verified to match.

This means the dataset metadata file is currently a runtime dependency of the inference system.

19. ML003 Inference Validation

ML003 has been verified for PyTorch/ONNX consistency.

Verified results:

Top-1 parity: 30/30
Top-3 parity: 30/30

Maximum probability difference:

0.00000086

Maximum logit difference:

0.00001287

The ONNX model therefore reproduces the verified PyTorch inference behavior within the tested tolerance.

20. ML003 Official Evaluation

The official ML003 test evaluation reports:

Test samples: 780
Classes: 78
Accuracy: 89.10%
Weighted precision: 90.63%
Weighted recall: 89.10%
Weighted F1: 88.65%

Best validation accuracy:

95.38%

Best validation epoch:

9

These metrics describe the evaluated ML model and should not be interpreted as a guarantee of real-world medical recognition accuracy.

21. Inference Abstraction

The backend contains a shared inference abstraction in:

backend/inference.py

It defines:

ModelInfo
InferenceResult
InferenceEngine

This abstraction allows the API/service layer to interact with the inference system without tightly coupling it to the implementation details of ONNX Runtime.

The file should not be removed simply because ML003 is currently the production model.

22. Prediction Service

The main business workflow is implemented in:

backend/services/prediction_service.py

Responsibilities include:

Image validation
Image metadata creation
ML inference
Medicine database lookup
Prediction persistence
Transaction handling
Logging
Rollback on failure

The service rolls back database changes if an exception occurs during prediction processing.

23. Database

The current development database is:

SQLite

Database file:

data/app.db

SQLAlchemy is used as the ORM.

The current database includes entities for:

Medicines
Model versions
Image metadata
Predictions

Foreign-key relationships are used to connect prediction records with relevant image, medicine, and model-version records.

24. Database Migrations

Database migration infrastructure is provided through:

Alembic

Configuration:

alembic.ini

Migration directory:

alembic/

Production database selection has not yet been finalized.

SQLite is currently suitable for development and testing. PostgreSQL should be evaluated before a multi-user production deployment.

25. Error Handling

The backend has a global exception handler.

Unexpected server-side exceptions return:

{
  "error": "internal_server_error",
  "message": "An unexpected error occurred."
}

HTTP status:

500

Internal exception details are logged server-side but are not exposed directly to API clients.

26. HTTP Error Behavior

Current behavior includes:

Situation	Status
Invalid uploaded image	400
Invalid image size/type	400
Missing required /predict file	422
Invalid /verify request	422
Malformed multipart request	400
Unexpected server exception	500

The current FastAPI validation behavior is intentional and has been tested.

27. Logging

The backend logs important operational events including:

Image validation success
Image metadata creation
Inference result
Model version
Medicine resolution
Prediction persistence
Transaction rollback
Unexpected exceptions

Uploaded filenames are not included in the image-validation log messages.

This reduces unnecessary exposure of user-provided filename information.

28. Security Considerations

Current protections include:

Maximum upload size
MIME type validation
Actual image validation using Pillow
Image dimension limits
Temporary file cleanup
Generic unexpected-error responses
No uploaded filename in validation logs
Pydantic request validation
Database transaction rollback on prediction failure

Additional production security hardening is still required before public deployment.

Planned security work includes:

Production CORS configuration
Security headers
Rate limiting/abuse protection where appropriate
Production secret management
Stronger request/resource controls
Production database configuration
Deployment-level security configuration
29. CORS

CORS configuration has not yet been finalized for production frontend deployment.

During frontend integration, CORS should be configured to allow only the required frontend origin(s).

Do not use unrestricted production CORS such as:

*

when credentials or restricted browser access are required.

The final CORS configuration should be environment-specific.

30. Configuration

Production configuration should not be hard-coded into application source code.

Recommended configuration categories include:

DATABASE_URL
FRONTEND_ORIGIN
ENVIRONMENT
LOG_LEVEL
MODEL_PATH
MAX_UPLOAD_SIZE

Secrets, credentials, API keys, and production database credentials must be supplied through secure environment configuration rather than committed to Git.

31. Dependency Management

Production runtime dependencies are stored in:

requirements.txt

Current pinned runtime dependencies include:

fastapi==0.142.2
uvicorn==0.54.0
SQLAlchemy==2.1.2
pydantic==2.13.5
python-multipart==0.0.32
Pillow==12.3.0
numpy==2.5.3
pandas==3.0.6
onnxruntime==1.30.0
torch==2.14.1
torchvision==0.29.1
opencv-python==5.0.0.93

Development/testing dependencies are stored separately in:

requirements-dev.txt

Current development dependencies include:

-r requirements.txt

pytest==9.1.1
httpx==0.28.1

This keeps the production dependency file separate from testing tools.

32. Clean Environment Verification

A separate clean virtual environment named:

.venv_clean

was used to verify dependency reproducibility.

The clean environment successfully:

Installed the runtime dependencies
Imported all required runtime packages
Ran the backend test suite
Loaded ML003
Started the FastAPI application
Performed real ML inference
Returned the expected prediction response

The environment is a verification environment and is not required to be committed to Git.

It may be removed after deployment preparation is complete.

33. Testing

Tests are located under:

tests/

Pytest discovery is restricted through:

pytest.ini

Current configuration:

[pytest]
testpaths = tests
python_files = test_*.py

This prevents archived and experimental scripts from being incorrectly collected as active tests.

34. Current Test Status

The current backend test suite has:

20 passed

The verified suite includes tests covering:

ML003 valid prediction
Invalid model output dimensions
Wrong class count
NaN logits
Positive infinity logits
Invalid batch size
Empty model output
Invalid medicine label
Prediction-service rollback behavior
API/backend behavior

Current result:

20 passed

There are currently non-blocking SQLAlchemy deprecation warnings related to datetime.utcnow().

These warnings do not currently cause test failures.

35. Verified End-to-End Prediction

A real handwritten medicine image was successfully processed using the clean environment.

Input:

data/metadata/Doctor’s Handwritten Prescription BD dataset/Testing/testing_words/0.png

Response:

{
  "medicine": "Aceta",
  "generic_name": "Paracetamol",
  "confidence": 0.9699714779853821
}

This verifies the complete backend workflow from image upload through ML inference and medicine lookup.

36. Health Verification

The backend has also been verified through:

curl http://127.0.0.1:8000/health

Successful response:

{
  "status": "ok",
  "database": "ok",
  "model": "ok"
}

This confirms that the database and ML003 inference engine were available to the running application.

37. Frontend Integration

The backend is currently suitable for beginning frontend integration.

The frontend should communicate with:

POST /predict

using:

multipart/form-data

with the uploaded image stored under:

file

Example frontend request conceptually:

POST /predict
Content-Type: multipart/form-data

file = handwritten-prescription-image

The frontend should display at least:

Predicted medicine
Generic name
Confidence

The frontend should also communicate clearly that the result is an AI/model prediction and should be verified by an appropriate medical professional.

38. Recommended Frontend Flow
User selects/takes image
        ↓
Frontend displays preview
        ↓
User submits image
        ↓
Frontend sends POST /predict
        ↓
Backend validates image
        ↓
ML003 performs inference
        ↓
Backend looks up medicine
        ↓
Backend returns JSON
        ↓
Frontend displays result
        ↓
User can verify/confirm the result
39. Frontend Error Handling

The frontend should handle at least:

400 → Invalid image
422 → Missing/invalid request data
500 → Backend/internal error
Network failure → Backend unavailable

The frontend should not assume that every /predict request succeeds.

40. API Documentation

FastAPI automatically provides OpenAPI documentation.

Swagger UI:

/docs

OpenAPI JSON:

/openapi.json

The Swagger documentation should be treated as the primary interactive API reference during frontend integration.

41. Production Readiness Status

The backend should currently be considered:

FRONTEND-INTEGRATION READY

but not yet:

FULL PRODUCTION DEPLOYMENT READY

The following major production tasks remain:

Production database evaluation
CORS configuration
Security hardening
Production configuration/environment management
Broader integration testing
Docker/containerization
Production deployment configuration
Final end-to-end acceptance testing
Monitoring/logging improvements
42. Current Backend Readiness Matrix
Area	Status
FastAPI application	Complete
/predict endpoint	Complete
/verify endpoint	Complete
/health endpoint	Complete
Image validation	Complete
ML003 ONNX integration	Complete
ONNX/PyTorch parity verification	Complete
Database integration	Complete for development
Prediction persistence	Complete
Transaction rollback	Complete
Global exception handling	Complete
Logging	Complete for current scope
Privacy/logging cleanup	Complete for current scope
Runtime dependency pinning	Complete
Development dependency separation	Complete
Clean environment verification	Complete
Frontend integration readiness	Ready
Production database	Pending
CORS production configuration	Pending
Security hardening	Pending
Docker	Pending
Production deployment configuration	Pending
Final production acceptance test	Pending
43. Important ML Safety Note

The system recognizes handwritten medicine names using a machine-learning model.

A model prediction is not equivalent to a medically confirmed prescription.

The backend therefore returns:

medicine
generic_name
confidence

but does not make prescribing, dosage, treatment, or clinical decisions.

The application should encourage appropriate verification by a qualified healthcare professional, especially when handwriting is ambiguous or the model confidence is low.

44. Development Commands
Activate development environment
source .venv/bin/activate
Install production dependencies
python -m pip install -r requirements.txt
Install development dependencies
python -m pip install -r requirements-dev.txt
Run tests
pytest -q
Start backend
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
Check health
curl http://127.0.0.1:8000/health
Open Swagger
http://127.0.0.1:8000/docs
45. Development Environment vs Production Environment

The current backend has been verified in a local development environment.

The following are development-oriented components:

SQLite
Local filesystem
Local ONNX model
Local dataset metadata
Local Uvicorn server

A production deployment should replace or configure these appropriately where required.

For example:

Development              Production
-----------              ----------
SQLite          →        PostgreSQL (evaluation required)
Local files     →        Proper model/data packaging
Local Uvicorn   →        Production container/server setup
Local config    →        Environment-based configuration
Development CORS→        Restricted production CORS
46. Deployment Philosophy

The ML003 model, preprocessing pipeline, and class mapping are considered verified components.

They should not be changed casually during backend hardening.

Any change to:

ML model
Preprocessing
Class mapping
ONNX model
Inference output

must be followed by appropriate regression and parity testing.

Backend infrastructure improvements should preserve the verified ML behavior.

47. Current Verified Baseline

The current verified backend baseline is:

Python: 3.12.7
FastAPI: 0.142.2
Uvicorn: 0.54.0
SQLAlchemy: 2.1.2
Pydantic: 2.13.5
Pillow: 12.3.0
NumPy: 2.5.3
ONNX Runtime: 1.30.0
PyTorch: 2.14.1
Torchvision: 0.29.1
OpenCV: 5.0.0.93
Pytest: 9.1.1
HTTPX: 0.28.1

ML baseline:

Model: ML003
Architecture: ResNet18
Format: ONNX
Runtime: ONNX Runtime
Input: 3 × 64 × 256
Classes: 78
Version: ML003-resnet18-onnx-1.0.0

Backend verification baseline:

Tests: 20 passed
Real /predict: Verified
/health: Verified
Clean environment: Verified
Runtime dependencies: Verified
48. Next Backend Development Tasks

After frontend integration begins, backend hardening should proceed in this order:

1. Production database evaluation
2. CORS configuration
3. Security hardening
4. Broader API/integration tests
5. Docker/containerization
6. Production configuration
7. Deployment configuration
8. Final end-to-end acceptance testing

These tasks should be performed incrementally so that the existing verified ML003 behavior is not accidentally changed.

49. Final Backend Status

Current status:

Backend Core:                 COMPLETE
ML003 Integration:            VERIFIED
API:                          VERIFIED
Database Integration:         VERIFIED
Testing:                      VERIFIED
Clean Environment:            VERIFIED
Frontend Integration:        READY
Production Hardening:        IN PROGRESS
Production Deployment:       NOT YET FINALIZED

The backend is therefore ready to proceed with frontend integration while production hardening continues in parallel.