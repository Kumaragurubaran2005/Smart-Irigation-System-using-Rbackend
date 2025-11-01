# run_api.R

library(plumber)

# Load the API
api <- plumb("api.R")

# Run the API
cat("Starting Water Requirement Prediction API...\n")
cat("API will be available at: http://localhost:8000\n")
cat("Endpoints:\n")
cat("  POST /predict - Make predictions\n")
cat("  GET  /health  - Health check\n")
cat("  GET  /model-info - Model information\n")

api$run(host = "0.0.0.0", port = 8000)