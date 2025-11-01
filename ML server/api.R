# api.R
library(plumber)
library(jsonlite)
library(dplyr)

library(httr)
library(jsonlite)

get_rainfall <- function(lat = 11.0, lon = 78.0) {
  url <- sprintf("https://api.open-meteo.com/v1/forecast?latitude=%f&longitude=%f&current=precipitation", lat, lon)
  
  tryCatch({
    response <- httr::GET(url)
    if (httr::status_code(response) == 200) {
      data <- jsonlite::fromJSON(httr::content(response, "text", encoding = "UTF-8"))
      rainfall <- data$current$precipitation
      if (is.null(rainfall)) rainfall <- 0
      return(as.numeric(rainfall))
    } else {
      cat("Rainfall API error:", httr::status_code(response), "\n")
      return(0)
    }
  }, error = function(e) {
    cat("Rainfall fetch error:", e$message, "\n")
    return(0)
  })
}

# Load the saved model and metadata
model <- readRDS("water_requirement_lightgbm_model.rds")
model_metadata <- readRDS("model_metadata.rds")

#* @apiTitle Water Requirement Prediction API
#* @apiDescription Predict water requirements for crops using LightGBM model

#* @filter cors
function(req, res) {
  res$setHeader("Access-Control-Allow-Origin", "*")
  res$setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
  res$setHeader("Access-Control-Allow-Headers", "Content-Type")
  
  if (req$REQUEST_METHOD == "OPTIONS") {
    res$status <- 200
    return(list())
  }
  
  forward()
}

#* Log information about the request
#* @filter logger
function(req) {
  cat("Request:", req$REQUEST_METHOD, req$PATH_INFO, "-", 
      as.character(Sys.time()), "\n")
  if (req$REQUEST_METHOD == "POST") {
    cat("Request body:", toJSON(req$postBody, auto_unbox = TRUE), "\n")
  }
  forward()
}

#* Predict water requirement
#* @param crop_planted The crop type
#* @param soil_type The soil type  
#* @param soil_moisture Soil moisture percentage
#* @param temperature Temperature in °C
#* @param humidity Humidity percentage
#* @param water_level Water level percentage
#* @param age_of_crop Age of crop in days
#* @post /predict
#* @serializer json
function(req, res, 
         crop_planted = "Wheat", 
         soil_type = "Loamy",
         soil_moisture = 0,
         temperature = 25,
         humidity = 50,
         water_level = 0,
         age_of_crop = 1) {
  
  tryCatch({
    # Parse JSON body if provided
    if (length(req$postBody) > 0) {
      body_data <- fromJSON(req$postBody)
      crop_planted <- body_data$crop_planted %||% crop_planted
      soil_type <- body_data$soil_type %||% soil_type
      soil_moisture <- as.numeric(body_data$soil_moisture %||% soil_moisture)
      temperature <- as.numeric(body_data$temperature %||% temperature)
      humidity <- as.numeric(body_data$humidity %||% humidity)
      water_level <- as.numeric(body_data$water_level %||% water_level)
      age_of_crop <- as.integer(body_data$age_of_crop %||% age_of_crop)
    }
    
    # Log input data
    cat("Processing prediction with parameters:\n")
    cat("  Crop:", crop_planted, "\n")
    cat("  Soil Type:", soil_type, "\n")
    cat("  Soil Moisture:", soil_moisture, "\n")
    cat("  Temperature:", temperature, "\n")
    cat("  Humidity:", humidity, "\n")
    cat("  Water Level:", water_level, "\n")
    cat("  Crop Age:", age_of_crop, "\n")
    rainfall = get_rainfall()
    # Transform frontend data to model format
    input_data <- data.frame(
      Crop.Age..days. = age_of_crop,
      Soil.Type = as.character(soil_type),
      Crop.Planted = as.character(crop_planted),
      Humidity.... = humidity,
      Temperature...C. = temperature,
      Water.Level..cm. = water_level,  # Note: Using water_level as cm
      Rainfall..mm. = rainfall
    )
    
    # Apply the same encoding as during training
    input_encoded <- input_data
    input_encoded$Soil.Type <- as.numeric(
      factor(input_encoded$Soil.Type, 
             levels = model_metadata$categorical_mapping$Soil.Type)
    ) - 1
    
    input_encoded$Crop.Planted <- as.numeric(
      factor(input_encoded$Crop.Planted, 
             levels = model_metadata$categorical_mapping$Crop.Planted)
    ) - 1
    
    # Ensure column order matches training
    input_encoded <- input_encoded[, model_metadata$feature_names]
    
    # Make prediction
    prediction <- predict(model, as.matrix(input_encoded))
    water_ml <- as.numeric(prediction[1])
    
    # Apply business rule override (similar to Python version)
    if (water_level > 15 && crop_planted != "Rice") {
      water_ml <- 0
      cat("Applied water level override\n")
    }
    
    # Prepare response
    response <- list(
      water_required = round(water_ml, 2),
      status = "success",
      message = "Prediction successful",
      features_used = list(
        crop_planted = crop_planted,
        soil_type = soil_type,
        soil_moisture = soil_moisture,
        temperature = temperature,
        humidity = humidity,
        water_level = water_level,
        age_of_crop = age_of_crop,
        rainfall = rainfall
      )
    )
    
    return(response)
    
  }, error = function(e) {
    cat("Prediction error:", e$message, "\n")
    res$status <- 400
    return(list(
      error = e$message,
      status = "error",
      message = "Prediction failed"
    ))
  })
}

#* Health check endpoint
#* @get /health
#* @serializer json
function() {
  return(list(
    status = "healthy",
    message = "Water Requirement Prediction API is running",
    model_loaded = !is.null(model),
    timestamp = as.character(Sys.time())
  ))
}

#* Get model info
#* @get /model-info
#* @serializer json
function() {
  return(list(
    model_type = "LightGBM",
    features = model_metadata$feature_names,
    categorical_mapping = model_metadata$categorical_mapping,
    performance = model_metadata$performance_metrics,
    training_date = model_metadata$training_date
  ))
}

# Helper function for NULL coalescing
`%||%` <- function(a, b) if (!is.null(a)) a else b