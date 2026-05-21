

setwd("C:/Users/sidha/Desktop/Classes/STA401/Datasets")  # update path as needed

# SECTION 0: LIBRARIES

library(readxl)
library(dplyr)
library(caret)
library(tree)
library(randomForest)
library(e1071)
library(xgboost)
library(class)
library(rpart)
library(rpart.plot)
library(smotefamily)
library(klaR)



# SECTION 1: LOAD DATA
df     <- read.csv("online_shoppers_intention.csv", stringsAsFactors = FALSE)
eda_df <- df   

str(df)
dim(df)


eda_df <- eda_df[!duplicated(eda_df), ]

# Fix data types for EDA
eda_df$Revenue    <- as.integer(eda_df$Revenue)
eda_df$Weekend    <- as.integer(eda_df$Weekend)
eda_df$SpecialDay <- as.numeric(eda_df$SpecialDay)
eda_df$Month      <- factor(eda_df$Month,
                            levels = c("Feb","Mar","Apr","May","Jun",
                                       "Jul","Aug","Sep","Oct","Nov","Dec"))
eda_df$VisitorType <- as.factor(eda_df$VisitorType)

dim(eda_df)   
str(eda_df)


# ── 2.1 Summary Statistics ───────────────────────────────────
summary(eda_df)


# ── 2.2 Target Variable: Revenue ─────────────────────────────
table(eda_df$Revenue)
prop.table(table(eda_df$Revenue))


barplot(table(eda_df$Revenue),
        main  = "Revenue Distribution",
        xlab  = "Revenue (0 = No Purchase, 1 = Purchase)",
        ylab  = "Number of Sessions",
        col   = c("steelblue", "darkorange"),
        names = c("No Purchase", "Purchase"))


# ── 2.3 Categorical Variables ─────────────────────────────────

# -- Month
table(eda_df$Month)


barplot(table(eda_df$Month),
        main = "Sessions by Month",
        xlab = "Month",
        ylab = "Number of Sessions",
        col  = "steelblue",
        las  = 2)

# -- VisitorType
table(eda_df$VisitorType)
prop.table(table(eda_df$VisitorType))


par(mfrow = c(1, 2))
barplot(table(eda_df$VisitorType),
        main = "Sessions by Visitor Type",
        xlab = "Visitor Type",
        ylab = "Number of Sessions",
        col  = c("steelblue", "darkorange", "darkgreen"),
        las  = 2)

# -- Weekend
table(eda_df$Weekend)
prop.table(table(eda_df$Weekend))


barplot(table(eda_df$Weekend),
        main  = "Weekend vs Weekday Sessions",
        xlab  = "Weekend (0 = Weekday, 1 = Weekend)",
        ylab  = "Number of Sessions",
        col   = c("steelblue", "darkorange"),
        names = c("Weekday", "Weekend"))

par(mfrow = c(2, 2))
barplot(table(eda_df$OperatingSystems), main = "Operating Systems",
        col = "steelblue",  xlab = "OS Code",     ylab = "Sessions")
barplot(table(eda_df$Browser),          main = "Browser Types",
        col = "darkorange", xlab = "Browser Code", ylab = "Sessions")
barplot(table(eda_df$Region),           main = "Regions",
        col = "darkgreen",  xlab = "Region Code",  ylab = "Sessions")
barplot(table(eda_df$TrafficType),      main = "Traffic Types",
        col = "purple",     xlab = "Traffic Code", ylab = "Sessions")
par(mfrow = c(1, 1))


par(mfrow = c(3, 3))
hist(eda_df$Administrative,
     main = "Administrative Pages",           xlab = "Count",   col = "steelblue",  breaks = 30)
hist(eda_df$Administrative_Duration,
     main = "Administrative Duration",        xlab = "Seconds", col = "steelblue",  breaks = 30)
hist(eda_df$Informational,
     main = "Informational Pages",            xlab = "Count",   col = "darkorange", breaks = 30)
hist(eda_df$Informational_Duration,
     main = "Informational Duration",         xlab = "Seconds", col = "darkorange", breaks = 30)
hist(eda_df$ProductRelated,
     main = "Product Related Pages",          xlab = "Count",   col = "darkgreen",  breaks = 30)
hist(eda_df$ProductRelated_Duration,
     main = "Product Related Duration",       xlab = "Seconds", col = "darkgreen",  breaks = 30)
hist(eda_df$BounceRates,
     main = "Bounce Rates",                   xlab = "Rate",    col = "purple",     breaks = 30)
hist(eda_df$ExitRates,
     main = "Exit Rates",                     xlab = "Rate",    col = "purple",     breaks = 30)
hist(eda_df$PageValues,
     main = "Page Values",                    xlab = "Value",   col = "red",        breaks = 30)
par(mfrow = c(1, 1))


pairs(Revenue ~ Administrative + Administrative_Duration +
        Informational + Informational_Duration +
        ProductRelated + ProductRelated_Duration +
        BounceRates + ExitRates + PageValues,
      data = eda_df,
      main = "Pairwise Relationships with Revenue")

num_cols <- c("Administrative", "Administrative_Duration",
              "Informational",  "Informational_Duration",
              "ProductRelated", "ProductRelated_Duration",
              "BounceRates",    "ExitRates",
              "PageValues",     "SpecialDay", "Revenue")

cor_matrix <- round(cor(eda_df[, num_cols]), 2)
print(cor_matrix)


# ── 2.7 Purchase Rate by Categorical Variables ───────────────


# SECTION 3: FEATURE SELECTION — PCA

# PCA is performed on SMOTE-balanced numeric predictors so that
# Chi-Square, ANOVA, and PCA reflect the same balanced data the
# classifiers are trained on.

# Prepare numeric-only data for SMOTE + PCA
pca_numeric_cols <- c("Administrative", "Administrative_Duration",
                      "Informational", "Informational_Duration",
                      "ProductRelated", "ProductRelated_Duration",
                      "BounceRates", "ExitRates", "PageValues",
                      "SpecialDay", "OperatingSystems", "Browser",
                      "Region", "TrafficType", "Weekend", "Revenue")

pca_input         <- df[!duplicated(df), pca_numeric_cols]
pca_input$Revenue <- as.integer(pca_input$Revenue)
pca_input$Weekend <- as.integer(pca_input$Weekend)
smote_pca_input         <- pca_input
smote_pca_input$Revenue <- as.factor(smote_pca_input$Revenue)

set.seed(1)
smote_pca_result <- SMOTE(X        = dplyr::select(smote_pca_input, -Revenue),
                          target   = smote_pca_input$Revenue,
                          K        = 5,
                          dup_size = 0)

smote_pca_df         <- smote_pca_result$data[, colnames(smote_pca_input)[colnames(smote_pca_input) != "Revenue"]]
smote_pca_df$Revenue <- as.factor(smote_pca_result$data[, ncol(smote_pca_result$data)])

cat("=== Class Balance After SMOTE (PCA Input) ===\n")
print(table(smote_pca_df$Revenue))
print(prop.table(table(smote_pca_df$Revenue)))

# ── 3.1 Fit PCA ──────────────────────────────────────────────
pca_feature_cols <- c("Administrative", "Administrative_Duration",
                      "Informational", "Informational_Duration",
                      "ProductRelated", "ProductRelated_Duration",
                      "BounceRates", "ExitRates", "PageValues",
                      "SpecialDay", "OperatingSystems", "Browser",
                      "Region", "TrafficType")

shoppers_numeric <- smote_pca_df[, pca_feature_cols]

apply(shoppers_numeric, 2, mean)
apply(shoppers_numeric, 2, sd)

pr.out <- prcomp(shoppers_numeric, scale = TRUE)

names(pr.out)
pr.out$center    # column means used for centering
pr.out$scale     # column SDs used for scaling
pr.out$rotation  # loadings — variable contributions to each PC

# ── 3.2 Eigenvalues ──────────────────────────────────────────
eigenvalues <- pr.out$sdev^2

cat("\n=== Eigenvalues (Variance per PC) ===\n")
print(round(eigenvalues, 4))
summary(pr.out)


# ── 3.3 PCA Visualizations ───────────────────────────────────
pve <- eigenvalues / sum(eigenvalues)

# Scree Plot — Eigenvalues (Kaiser rule: keep PCs with eigenvalue > 1)
plot(eigenvalues,
     xlab = "Principal Component", ylab = "Eigenvalue",
     ylim = c(0, max(eigenvalues) + 0.5),
     type = "b", pch = 16, col = "steelblue",
     main = "Scree Plot — Eigenvalues")
abline(h = 1, col = "red", lty = 2, lwd = 2)
legend("topright", legend = "Eigenvalue = 1 (Kaiser Rule)", col = "red", lty = 2, lwd = 2)

# Scree Plot — Proportion of Variance Explained
plot(pve,
     xlab = "Principal Component", ylab = "Proportion of Variance Explained",
     ylim = c(0, 1), type = "b", pch = 16, col = "steelblue",
     main = "Scree Plot — Proportion of Variance Explained")

# Cumulative Variance Plot
plot(cumsum(pve),
     xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained",
     ylim = c(0, 1), type = "b", pch = 16, col = "darkorange",
     main = "Cumulative Variance Explained")
abline(h = 0.80, col = "red",       lty = 2, lwd = 2)
abline(h = 0.90, col = "darkgreen", lty = 2, lwd = 2)
legend("bottomright",
       legend = c("80% Threshold", "90% Threshold"),
       col    = c("red", "darkgreen"), lty = 2, lwd = 2)

# Biplot — Variable Loadings on PC1 vs PC2
plot(pr.out$rotation[, 1], pr.out$rotation[, 2],
     type = "n", xlab = "PC1", ylab = "PC2",
     main = "Biplot — Variable Loadings on PC1 vs PC2")
abline(h = 0, v = 0, lty = 2, col = "grey")
arrows(0, 0, pr.out$rotation[, 1], pr.out$rotation[, 2],
       length = 0.1, col = "red", lwd = 2)
text(pr.out$rotation[, 1], pr.out$rotation[, 2],
     labels = rownames(pr.out$rotation), cex = 0.8, pos = 3, col = "black")

# ── 3.4 Top Loadings ─────────────────────────────────────────
cat("\n=== Loadings: PC1 and PC2 ===\n")
print(round(pr.out$rotation[, 1:2], 4))

cat("\n=== Top Contributors to PC1 (sorted by absolute loading) ===\n")
print(round(sort(abs(pr.out$rotation[, 1]), decreasing = TRUE), 4))

cat("\n=== Top Contributors to PC2 (sorted by absolute loading) ===\n")
print(round(sort(abs(pr.out$rotation[, 2]), decreasing = TRUE), 4))

#After PCA analysis, we remove certain columns
cols_to_remove <- c("Region", "Browser", "OperatingSystems", "TrafficType")
df <- df[, !(names(df) %in% cols_to_remove)]

df$Revenue     <- as.integer(df$Revenue)
df$Weekend     <- as.integer(df$Weekend)
df$Month       <- as.factor(df$Month)
df$VisitorType <- as.factor(df$VisitorType)

numeric_cols <- c("Administrative", "Administrative_Duration", "Informational",
                  "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
                  "BounceRates", "ExitRates", "PageValues", "SpecialDay")
df[numeric_cols] <- lapply(df[numeric_cols], as.numeric)

sum(is.na(df))

# Remove Duplicates
sum(duplicated(df))
df <- df[!duplicated(df), ]
dim(df)

summary(df)
table(df$Revenue)
prop.table(table(df$Revenue))   # ~15.4% purchases — class imbalance confirmed
table(df$Month)
table(df$VisitorType)


# SECTION 5: DATA TRANSFORMATION


# ── 5.1 Log Transformation (right-skewed duration/count features) ─
df$log_Administrative_Duration <- log1p(df$Administrative_Duration)
df$log_Informational_Duration  <- log1p(df$Informational_Duration)
df$log_ProductRelated_Duration <- log1p(df$ProductRelated_Duration)
df$log_ProductRelated          <- log1p(df$ProductRelated)


# ── 5.2 One-Hot Encoding ─────────────────────────────────────
month_dummies   <- model.matrix(~ Month - 1, data = df)
visitor_dummies <- model.matrix(~ VisitorType - 1, data = df)
colnames(month_dummies)   <- gsub("Month", "Month_",             colnames(month_dummies))
colnames(visitor_dummies) <- gsub("VisitorType", "VisitorType_", colnames(visitor_dummies))

df <- cbind(df, month_dummies, visitor_dummies)
df <- dplyr::select(df, -c(Month, VisitorType))
dim(df)


# ── 5.3 Feature Scaling — Z-Score (for KNN) ──────────────────
scale_cols <- c("BounceRates", "ExitRates", "PageValues",
                "Administrative_Duration", "Informational_Duration",
                "ProductRelated_Duration")

df_scaled <- df
for (col in scale_cols) {
  new_col <- paste0("scaled_", col)
  df_scaled[[new_col]] <- as.numeric(scale(df_scaled[[col]]))
}


# ── 5.4 SMOTE — Fix Class Imbalance ──────────────────────────
# Applied after all cleaning and transformation; used for classification only.

set.seed(1)
smote_result <- SMOTE(X        = dplyr::select(df, -Revenue),
                      target   = as.factor(df$Revenue),
                      K        = 5,
                      dup_size = 0)   # dup_size=0 auto-balances to 50/50

smote_df         <- smote_result$data[, colnames(df)[colnames(df) != "Revenue"]]
smote_df$Revenue <- as.factor(smote_result$data[, ncol(smote_result$data)])

table(smote_df$Revenue)
prop.table(table(smote_df$Revenue))   # approx 50/50

# SMOTE for scaled version (used by KNN)
set.seed(1)
smote_scaled_result <- SMOTE(X        = dplyr::select(df_scaled, -Revenue),
                             target   = as.factor(df_scaled$Revenue),
                             K        = 5,
                             dup_size = 0)

smote_scaled         <- smote_scaled_result$data[, colnames(df_scaled)[colnames(df_scaled) != "Revenue"]]
smote_scaled$Revenue <- as.factor(smote_scaled_result$data[, ncol(smote_scaled_result$data)])

table(smote_scaled$Revenue)


# ── 5.5 PCA Transformation — Top 5 PCs ───────────────────────
# Based on scree plots above, the top 5 PCs are retained.
pca_prep <- smote_df %>% dplyr::select(-Revenue)
pr_out   <- prcomp(pca_prep, scale = TRUE)

pc_df         <- as.data.frame(pr_out$x[, 1:5])
pc_df$Revenue <- smote_df$Revenue


# SECTION 6: DATA PARTITIONING

set.seed(123)
train_index <- createDataPartition(smote_df$Revenue, p = 0.70, list = FALSE)

# Standard partitions
train_df <- smote_df[train_index, ]
test_df  <- smote_df[-train_index, ]

# Scaled partitions (KNN)
train_scaled <- smote_scaled[train_index, ]
test_scaled  <- smote_scaled[-train_index, ]

# PCA partitions
train_pc <- pc_df[train_index, ]
test_pc  <- pc_df[-train_index, ]

cat("Train size:", nrow(train_df), "\n")
cat("Test size :", nrow(test_df),  "\n")



# SECTION 7: MODEL TRAINING & EVALUATION


# ── 7.1 Logistic Regression ──────────────────────────────────
cat("\n--- 1. LOGISTIC REGRESSION ---\n")

glm_model <- glm(Revenue ~ ., data = train_df, family = "binomial")
summary(glm_model)

glm_probs <- predict(glm_model, newdata = test_df, type = "response")
glm_pred  <- as.factor(ifelse(glm_probs > 0.5, 1, 0))
glm_cm    <- confusionMatrix(glm_pred, test_df$Revenue)
print(glm_cm)


# ── 7.2 Decision Tree (Pruned) ───────────────────────────────
cat("\n--- 2. DECISION TREE (PRUNED) ---\n")

tree_model <- tree(Revenue ~ ., data = train_df)

set.seed(123)
cv_tree <- cv.tree(tree_model, FUN = prune.misclass)
plot(cv_tree$size, cv_tree$dev, type = "b",
     main = "CV: Deviance vs Tree Size", xlab = "Tree Size", ylab = "Deviance")

optimal_size <- cv_tree$size[which.min(cv_tree$dev)]
cat("Optimal Tree Size:", optimal_size, "\n")

pruned_tree <- prune.misclass(tree_model, best = optimal_size)
plot(pruned_tree)
text(pruned_tree, pretty = 0)

tree_pred <- predict(pruned_tree, test_df, type = "class")
tree_cm   <- confusionMatrix(tree_pred, test_df$Revenue)
print(tree_cm)


# ── 7.3 Random Forest ────────────────────────────────────────
cat("\n--- 3. RANDOM FOREST ---\n")

set.seed(123)
rf_model <- randomForest(Revenue ~ ., data = train_df, ntree = 500, importance = TRUE)
print(rf_model)

rf_pred <- predict(rf_model, newdata = test_df)
rf_cm   <- confusionMatrix(rf_pred, test_df$Revenue)
print(rf_cm)

varImpPlot(rf_model, main = "Random Forest: Variable Importance")


# ── 7.4 XGBoost ──────────────────────────────────────────────
cat("\n--- 4. XGBOOST ---\n")

train_x <- as.matrix(dplyr::select(train_df, -Revenue)); train_y <- as.numeric(as.character(train_df$Revenue))
test_x  <- as.matrix(dplyr::select(test_df,  -Revenue)); test_y  <- as.numeric(as.character(test_df$Revenue))

dtrain    <- xgb.DMatrix(data = train_x, label = train_y)
xgb_model <- xgb.train(
  params  = list(objective = "binary:logistic", eta = 0.3),
  data    = dtrain,
  nrounds = 50
)

xgb_probs <- predict(xgb_model, test_x)
xgb_pred  <- as.factor(ifelse(xgb_probs > 0.5, 1, 0))
xgb_cm    <- confusionMatrix(xgb_pred, as.factor(test_y))
print(xgb_cm)


# ── 7.5 KNN ──────────────────────────────────────────────────
cat("\n--- 5. K-NEAREST NEIGHBORS (KNN) ---\n")

tct <- trainControl(method = "cv", number = 10)
knn_tuning <- train(Revenue ~ .,
                    data       = train_scaled %>% dplyr::select(Revenue, contains("scaled_")),
                    method     = "knn",
                    trControl  = tct,
                    tuneLength = 15)
print(knn_tuning)
plot(knn_tuning)

best_k <- knn_tuning$bestTune$k
cat("Best K:", best_k, "\n")

knn_train_x <- train_scaled %>% dplyr::select(contains("scaled_"))
knn_test_x  <- test_scaled  %>% dplyr::select(contains("scaled_"))

knn_pred <- knn(train = knn_train_x, test = knn_test_x,
                cl = train_scaled$Revenue, k = best_k)
knn_cm   <- confusionMatrix(knn_pred, test_scaled$Revenue)
print(knn_cm)


# ── 7.6 Naive Bayes ──────────────────────────────────────────
cat("\n--- 6. NAIVE BAYES ---\n")

nb_model <- naiveBayes(Revenue ~ ., data = train_df)
nb_pred  <- predict(nb_model, newdata = test_df)
nb_cm    <- confusionMatrix(nb_pred, test_df$Revenue)
print(nb_cm)


# ── 7.7 PCA-Based Logistic Regression (Top 5 PCs) ────────────
cat("\n--- 7. PCA-BASED LOGISTIC REGRESSION ---\n")

pca_glm   <- glm(Revenue ~ ., data = train_pc, family = "binomial")
pca_probs <- predict(pca_glm, newdata = test_pc, type = "response")
pca_pred  <- as.factor(ifelse(pca_probs > 0.5, 1, 0))
pca_cm    <- confusionMatrix(pca_pred, test_pc$Revenue)
print(pca_cm)


# SECTION 8: FINAL MODEL COMPARISON

cat("\n--- FINAL MODEL COMPARISON ---\n")

cm_list <- list(
  "Logit" = glm_cm, "Tree" = tree_cm, "RF" = rf_cm, 
  "XGB" = xgb_cm, "KNN" = knn_cm, "NB" = nb_cm, "PCA_Logit" = pca_cm
)

# Extracting metrics into a clean table
results <- data.frame(
  Model       = names(cm_list),
  Accuracy    = sapply(cm_list, function(x) x$overall["Accuracy"]),
  Sensitivity = sapply(cm_list, function(x) x$byClass["Sensitivity"]),
  F1_Score    = sapply(cm_list, function(x) x$byClass["F1"])
)

# Sort by F1 Score (usually best for imbalanced business cases)
results_sorted <- results[order(-results$F1_Score), ]
print(results_sorted)
