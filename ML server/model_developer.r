# Load required libraries
library(lightgbm)
library(dplyr)
library(caret)
library(ggplot2)

# Read the data
data <- read.csv("synthetic_water_requirement_data.csv")

# Display basic info
cat("Dataset dimensions:", dim(data), "\n")
cat("\nFirst few rows:\n")
print(head(data))
cat("\nStructure:\n")
str(data)

# Preprocessing: Convert categorical variables to factors
data$Soil.Type <- as.factor(data$Soil.Type)
data$Crop.Planted <- as.factor(data$Crop.Planted)

# Prepare features and target
X <- data %>% select(-Water.Required..ml.)
y <- data$Water.Required..ml.

# Convert factors to numeric for LightGBM
X_encoded <- X
X_encoded$Soil.Type <- as.numeric(X_encoded$Soil.Type) - 1  # LightGBM expects 0-based categories
X_encoded$Crop.Planted <- as.numeric(X_encoded$Crop.Planted) - 1

# Identify categorical features for LightGBM
categorical_features <- which(names(X_encoded) %in% c("Soil.Type", "Crop.Planted")) - 1  # 0-based indexing

# Split the data
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X_encoded[train_index, ]
X_test <- X_encoded[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

cat("\nTraining set size:", dim(X_train), "\n")
cat("Test set size:", dim(X_test), "\n")

# Create LightGBM datasets
dtrain <- lgb.Dataset(
  data = as.matrix(X_train),
  label = y_train,
  categorical_feature = categorical_features
)

dtest <- lgb.Dataset(
  data = as.matrix(X_test),
  label = y_test,
  categorical_feature = categorical_features
)

# Define LightGBM parameters
params <- list(
  objective = "regression",
  metric = "rmse",
  learning_rate = 0.05,
  num_leaves = 31,
  max_depth = -1,
  min_data_in_leaf = 20,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  bagging_freq = 5,
  verbosity = -1,
  seed = 42
)

# Train the model
cat("\nTraining LightGBM model...\n")
model <- lgb.train(
  params = params,
  data = dtrain,
  valids = list(test = dtest),
  nrounds = 1000,
  early_stopping_rounds = 50,
  verbose = 1,
  eval_freq = 100
)

# Make predictions
y_pred <- predict(model, as.matrix(X_test))

# Calculate metrics
mae <- mean(abs(y_test - y_pred))
mse <- mean((y_test - y_pred)^2)
rmse <- sqrt(mse)
r2 <- cor(y_test, y_pred)^2

cat("\n", rep("=", 50), "\n")
cat("MODEL PERFORMANCE METRICS\n")
cat(rep("=", 50), "\n")
cat("Mean Absolute Error (MAE):", round(mae, 4), "\n")
cat("Mean Squared Error (MSE):", round(mse, 4), "\n")
cat("Root Mean Squared Error (RMSE):", round(rmse, 4), "\n")
cat("R² Score:", round(r2, 4), "\n")

# Feature importance
feature_importance <- lgb.importance(model, percentage = TRUE)
cat("\n", rep("=", 50), "\n")
cat("FEATURE IMPORTANCE\n")
cat(rep("=", 50), "\n")
print(feature_importance)

# Save the model as RDS
model_filename <- "water_requirement_lightgbm_model.rds"
saveRDS(model, model_filename)
cat("\nModel saved as:", model_filename, "\n")

# Save feature encodings and metadata for future use
model_metadata <- list(
  feature_names = names(X_encoded),
  categorical_mapping = list(
    Soil.Type = levels(X$Soil.Type),
    Crop.Planted = levels(X$Crop.Planted)
  ),
  training_date = Sys.Date(),
  performance_metrics = list(MAE = mae, MSE = mse, RMSE = rmse, R2 = r2)
)

saveRDS(model_metadata, "model_metadata.rds")
cat("Model metadata saved as: model_metadata.rds\n")

# Visualization
# 1. Feature Importance Plot
feature_imp_plot <- ggplot(feature_importance, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance", x = "Features", y = "Importance (Gain)") +
  theme_minimal()

# 2. Actual vs Predicted Plot
results_df <- data.frame(Actual = y_test, Predicted = y_pred)
actual_vs_pred_plot <- ggplot(results_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Actual vs Predicted", x = "Actual Water Required", y = "Predicted Water Required") +
  theme_minimal()

# 3. Residuals Plot
results_df$Residuals <- results_df$Actual - results_df$Predicted
residuals_plot <- ggplot(results_df, aes(x = Predicted, y = Residuals)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residuals Plot", x = "Predicted Values", y = "Residuals") +
  theme_minimal()

# Display plots
print(feature_imp_plot)
print(actual_vs_pred_plot)
print(residuals_plot)

# Sample predictions
cat("\n", rep("=", 50), "\n")
cat("SAMPLE PREDICTIONS\n")
cat(rep("=", 50), "\n")
set.seed(123)
sample_indices <- sample(1:length(y_test), 10)
for (i in seq_along(sample_indices)) {
  idx <- sample_indices[i]
  actual <- y_test[idx]
  predicted <- y_pred[idx]
  error <- abs(actual - predicted)
  cat(sprintf("Sample %d: Actual = %.2f, Predicted = %.2f, Error = %.2f\n", 
              i, actual, predicted, error))
}

# Model summary
cat("\n", rep("=", 50), "\n")
cat("MODEL SUMMARY\n")
cat(rep("=", 50), "\n")
cat("Best iteration:", model$best_iter, "\n")
cat("Best score:", model$best_score, "\n")
cat("Number of features:", length(model$feature_name), "\n")