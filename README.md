# Palmer Penguins Species Prediction API

A production-ready REST API that predicts penguin species (**Adelie**, **Chinstrap**, or **Gentoo**) based on anatomical measurements. This project serves as a clean template for deploying scikit-learn models using **FastAPI** and **Docker**.



##  Features

- **FastAPI Framework**: High-performance, easy-to-use REST API.
- **Random Forest Classifier**: Robust machine learning model with high accuracy (~98%).
- **Automated Docs**: Interactive API documentation via Swagger UI.
- **Validation**: Strict Pydantic schemas to ensure anatomical measurements are realistic.
- **Dockerized**: Containerized environment for consistent deployment.
- **CI/CD Friendly**: Model training is integrated into the build process.

---

##  Model Details

The model is trained on the [Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/), using four key features:

| Feature | Description | Range (Approx) |
| :--- | :--- | :--- |
| `bill_length_mm` | Length of the penguin's beak (culmen) | 32mm - 60mm |
| `bill_depth_mm` | Vertical thickness of the beak | 13mm - 22mm |
| `flipper_length_mm` | Length of the wing | 170mm - 235mm |
| `body_mass_g` | Total body mass in grams | 2700g - 6300g |



---

##  Installation & Setup

### 1. Local Development
Clone the repository and set up a virtual environment:

```bash
# Clone the repo
git clone https://github.com/T-Luxshan/Palmer-Penguins-api.git
cd palmer-penguins-api

# Install dependencies
pip install -r requirements.txt

# Train the model (generates penguin_model.pkl)
python train.py

# Launch the API
uvicorn main:app --reload

### 2. Running with Docker
The Dockerfile is configured to train the model during the image build process.
````

### Dockerization
```bash
# Build the image
docker build -t penguins-api .

# Run the container
docker run -d -p 8000:8000 --name penguins-api-container penguins-api
```

### API Usage
Once the server is running, navigate to:

- Interactive Swagger UI: http://localhost:8000/docs

- Health Check: http://localhost:8000/

### Example request body (JSON)

```json
{
  "bill_length_mm": 39.5,
  "bill_depth_mm": 17.4,
  "flipper_length_mm": 186.0,
  "body_mass_g": 3800.0
}
```

### Model Performance

- Dataset: Palmer Penguins (~344 rows, cleaned to ~333 valid samples)
- Model: RandomForestClassifier (100 estimators, random_state=42)
- Features: bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
- Target: species (3 classes: Adelie, Chinstrap, Gentoo)
- Typical performance: ~97–99% accuracy (excellent separation of classes)

### Tech Stack

- Python 3.11+
- FastAPI
- scikit-learn
- palmerpenguins
- Pydantic
- Docker

