# test_api.R
library(httr)
library(jsonlite)

# Test the prediction endpoint
test_data <- list(
  crop_planted = "Wheat",
  soil_type = "Loamy", 
  soil_moisture = 0,
  temperature = 25,
  humidity = 50,
  water_level = 0,
  age_of_crop = 100
)

response <- POST(
  "http://localhost:8000/predict",
  body = toJSON(test_data, auto_unbox = TRUE),
  content_type("application/json")
)

cat("Status:", response$status_code, "\n")
cat("Response:\n")
print(fromJSON(content(response, "text")))
