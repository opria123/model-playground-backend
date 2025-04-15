# Model Playground Backend

A Flask API backend service for model playground.

## Running Locally

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## Running with Docker

1. Build the Docker image:
```bash
docker build -t model-playground-backend .
```

2. Run the container with live reloading:

For Linux/Mac:
```bash
docker run -p 5000:5000 -v $(pwd):/app model-playground-backend
```

For Windows PowerShell:
```powershell
docker run -p 5000:5000 -v ${PWD}:/app model-playground-backend
```

Note: The `-v` flag mounts your local directory to the container, enabling live reloading of changes.

## API Endpoints

- `GET /health` - Health check endpoint
- `GET /api/hello` - Sample hello world endpoint
- `POST /api/generate` - Generate text using the model
  - Request body:
    ```json
    {
        "prompt": "Your prompt here",
        "role": "assistant",  // optional, defaults to "assistant"
        "max_tokens": 256     // optional, defaults to 256
    }
    ```
  - Response:
    ```json
    {
        "response": "Generated text from the model"
    }
    ```
