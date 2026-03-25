##########################
# STATISTICS AND BIG DATA II - GROUP PROJECT
# Predicting Airbus Weekly Returns
##########################

# ============================================
# INSTALL AND LOAD ALL PACKAGES
# ============================================
install.packages(c("readxl", "corrplot", "ggplot2", "reshape2", "moments",
                   "car", "glmnet", "gam", "randomForest", "gbm",
                   "kableExtra", "knitr", "lmtest"),
                 repos = "https://cran.r-project.org")

library(readxl)
library(corrplot)
library(ggplot2)
library(reshape2)
library(moments)
library(car)
library(glmnet)
library(gam)
library(randomForest)
library(gbm)
library(kableExtra)
library(knitr)
library(lmtest)

# ============================================
# PART 1: DATA DESCRIPTION
# ============================================

# ---- 1.1 Load Data ----
library(readxl)
Airbus_Database <- read_excel("Desktop/Projet STAT BIG DATA 2/RENDU FINAL/Airbus_Database.xlsx")
View(Airbus_Database)
colnames(data) <- trimws(colnames(data))
data$Date <- as.Date(data$Date)
returns <- data[, -1]

str(data)
head(data)
dim(data)

# ---- 1.2 Descriptive Statistics ----
desc_stats <- data.frame(
  Mean     = apply(returns, 2, mean, na.rm = TRUE),
  Median   = apply(returns, 2, median, na.rm = TRUE),
  Std.Dev  = apply(returns, 2, sd, na.rm = TRUE),
  Min      = apply(returns, 2, min, na.rm = TRUE),
  Max      = apply(returns, 2, max, na.rm = TRUE),
  Skewness = apply(returns, 2, skewness, na.rm = TRUE),
  Kurtosis = apply(returns, 2, kurtosis, na.rm = TRUE)
)
print(round(desc_stats, 6))

desc_stats_round <- round(desc_stats, 4)
kbl(desc_stats_round, format = "html",
    caption = "Table 1: Descriptive Statistics of Weekly Returns (Jan 2016 - Jan 2026)",
    align = "c") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"),
                full_width = FALSE, font_size = 14) %>%
  row_spec(0, bold = TRUE, background = "#2F5496", color = "white") %>%
  column_spec(1, bold = TRUE)

# ---- 1.3 Correlation Matrix ----
cor_matrix <- cor(returns, use = "complete.obs")
print(round(cor_matrix, 4))

par(mfrow = c(1, 1))
corrplot(cor_matrix, method = "color", type = "upper", order = "hclust",
         tl.col = "black", tl.cex = 0.9, addCoef.col = "black", number.cex = 0.7,
         col = colorRampPalette(c("#BB4444", "#FFFFFF", "#4477AA"))(200),
         title = "Correlation Matrix - Weekly Returns", mar = c(0, 0, 2, 0))

# ---- 1.4 Returns Multi-Panel (2x2) ----
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
plot(data$Date, data$AIR, type = "l", col = "darkblue", lwd = 0.8,
     main = "Airbus (AIR FP)", xlab = "", ylab = "Weekly Return")
abline(h = 0, col = "red", lty = 2)
plot(data$Date, data$BA, type = "l", col = "darkred", lwd = 0.8,
     main = "Boeing (BA US)", xlab = "", ylab = "Weekly Return")
abline(h = 0, col = "red", lty = 2)
plot(data$Date, data$CAC, type = "l", col = "darkgreen", lwd = 0.8,
     main = "CAC 40 Index", xlab = "Date", ylab = "Weekly Return")
abline(h = 0, col = "red", lty = 2)
plot(data$Date, data$VIX, type = "l", col = "purple", lwd = 0.8,
     main = "VIX Index", xlab = "Date", ylab = "Weekly Return")
abline(h = 0, col = "red", lty = 2)

# ---- 1.5 Distribution of Airbus Returns ----
par(mfrow = c(1, 1))
hist(data$AIR, breaks = 40, col = "steelblue", border = "white",
     main = "Distribution of Airbus Weekly Returns",
     xlab = "Return", probability = TRUE)
curve(dnorm(x, mean = mean(data$AIR), sd = sd(data$AIR)),
      add = TRUE, col = "red", lwd = 2)
legend("topright", legend = "Normal fit", col = "red", lwd = 2)

# ============================================
# PART 2: MODELING AND FORECASTING
# ============================================

# ---- 2.1 Train / Test Split (80/20) ----
train <- data[data$Date < as.Date("2024-01-01"), ]
test  <- data[data$Date >= as.Date("2024-01-01"), ]
p <- 8
n.train <- nrow(train)
n.test  <- nrow(test)
cat("Train set:", n.train, "obs | Test set:", n.test, "obs\n")

# ---- 2.2 MODEL 1 - OLS ----
ols.fit <- lm(AIR ~ BA + SAF + CAC + EURUSD + VIX + BRENT + US10YR + GE, data = train)
summary(ols.fit)
vif(ols.fit)

ols.pred <- predict(ols.fit, newdata = test)
ols.mse  <- mean((test$AIR - ols.pred)^2)
ols.rmse <- sqrt(ols.mse)
ols.mae  <- mean(abs(test$AIR - ols.pred))
cat("\n--- OLS ---\nMSE:", round(ols.mse,6), "| RMSE:", round(ols.rmse,6), "| MAE:", round(ols.mae,6), "\n")

par(mfrow = c(1, 1))
plot(test$Date, test$AIR, type = "l", col = "black", lwd = 1.5,
     main = "OLS: Actual vs Predicted", xlab = "Date", ylab = "Weekly Return")
lines(test$Date, ols.pred, col = "blue", lwd = 1.5, lty = 2)
abline(h = 0, col = "gray", lty = 3)
legend("topright", legend = c("Actual", "OLS"), col = c("black", "blue"), lty = c(1, 2), lwd = 1.5)

plot(ols.pred, test$AIR, pch = 20, col = "blue",
     main = "OLS: Scatter Plot", xlab = "Predicted", ylab = "Actual")
abline(0, 1, col = "red", lwd = 2)

# ---- 2.3 MODEL 2 - LASSO ----
x.train <- as.matrix(train[, c("BA","SAF","CAC","EURUSD","VIX","BRENT","US10YR","GE")])
y.train <- train$AIR
x.test  <- as.matrix(test[, c("BA","SAF","CAC","EURUSD","VIX","BRENT","US10YR","GE")])
y.test  <- test$AIR

set.seed(1)
cv.lasso <- cv.glmnet(x.train, y.train, alpha = 1)
plot(cv.lasso)
bestlam <- cv.lasso$lambda.min

lasso.mod  <- glmnet(x.train, y.train, alpha = 1)
lasso.coef <- predict(lasso.mod, type = "coefficients", s = bestlam)
print(lasso.coef)

lasso.pred <- predict(lasso.mod, s = bestlam, newx = x.test)
lasso.mse  <- mean((y.test - lasso.pred)^2)
lasso.rmse <- sqrt(lasso.mse)
lasso.mae  <- mean(abs(y.test - lasso.pred))
cat("\n--- Lasso ---\nMSE:", round(lasso.mse,6), "| RMSE:", round(lasso.rmse,6), "| MAE:", round(lasso.mae,6), "\n")

par(mfrow = c(1, 1))
plot(test$Date, test$AIR, type = "l", col = "black", lwd = 1.5,
     main = "Lasso: Actual vs Predicted", xlab = "Date", ylab = "Weekly Return")
lines(test$Date, lasso.pred, col = "red", lwd = 1.5, lty = 2)
abline(h = 0, col = "gray", lty = 3)
legend("topright", legend = c("Actual", "Lasso"), col = c("black", "red"), lty = c(1, 2), lwd = 1.5)

plot(lasso.pred, test$AIR, pch = 20, col = "red",
     main = "Lasso: Scatter Plot", xlab = "Predicted", ylab = "Actual")
abline(0, 1, col = "red", lwd = 2)

# ---- 2.4 MODEL 3 - GAM ----
gam.fit <- gam(AIR ~ s(BA,4) + s(SAF,4) + s(CAC,4) + s(EURUSD,4) +
               s(VIX,4) + s(BRENT,4) + s(US10YR,4) + s(GE,4), data = train)
summary(gam.fit)

par(mfrow = c(2, 4), mar = c(4, 3, 2, 1))
plot(gam.fit, se = TRUE, col = "blue")

gam.pred <- predict(gam.fit, newdata = test)
gam.mse  <- mean((test$AIR - gam.pred)^2)
gam.rmse <- sqrt(gam.mse)
gam.mae  <- mean(abs(test$AIR - gam.pred))
cat("\n--- GAM ---\nMSE:", round(gam.mse,6), "| RMSE:", round(gam.rmse,6), "| MAE:", round(gam.mae,6), "\n")

par(mfrow = c(1, 1))
plot(test$Date, test$AIR, type = "l", col = "black", lwd = 1.5,
     main = "GAM: Actual vs Predicted", xlab = "Date", ylab = "Weekly Return")
lines(test$Date, gam.pred, col = "darkgreen", lwd = 1.5, lty = 2)
abline(h = 0, col = "gray", lty = 3)
legend("topright", legend = c("Actual", "GAM"), col = c("black", "darkgreen"), lty = c(1, 2), lwd = 1.5)

plot(gam.pred, test$AIR, pch = 20, col = "darkgreen",
     main = "GAM: Scatter Plot", xlab = "Predicted", ylab = "Actual")
abline(0, 1, col = "red", lwd = 2)

# ---- 2.5 MODEL 4 - BAGGING ----
set.seed(1)
bag.fit <- randomForest(AIR ~ BA + SAF + CAC + EURUSD + VIX + BRENT + US10YR + GE,
                        data = train, mtry = 8, ntree = 500, importance = TRUE)
bag.fit

bag.pred <- predict(bag.fit, newdata = test)
bag.mse  <- mean((test$AIR - bag.pred)^2)
bag.rmse <- sqrt(bag.mse)
bag.mae  <- mean(abs(test$AIR - bag.pred))
cat("\n--- Bagging ---\nMSE:", round(bag.mse,6), "| RMSE:", round(bag.rmse,6), "| MAE:", round(bag.mae,6), "\n")

par(mfrow = c(1, 1))
plot(test$Date, test$AIR, type = "l", col = "black", lwd = 1.5,
     main = "Bagging: Actual vs Predicted", xlab = "Date", ylab = "Weekly Return")
lines(test$Date, bag.pred, col = "orange", lwd = 1.5, lty = 2)
abline(h = 0, col = "gray", lty = 3)
legend("topright", legend = c("Actual", "Bagging"), col = c("black", "orange"), lty = c(1, 2), lwd = 1.5)

plot(bag.pred, test$AIR, pch = 20, col = "orange",
     main = "Bagging: Scatter Plot", xlab = "Predicted", ylab = "Actual")
abline(0, 1, col = "red", lwd = 2)

# ---- 2.6 MODEL 5 - RANDOM FOREST ----
set.seed(1)
rf.fit <- randomForest(AIR ~ BA + SAF + CAC + EURUSD + VIX + BRENT + US10YR + GE,
                       data = train, mtry = 3, ntree = 500, importance = TRUE)
rf.fit

par(mfrow = c(1, 1))
varImpPlot(rf.fit, main = "Random Forest: Variable Importance")

rf.pred <- predict(rf.fit, newdata = test)
rf.mse  <- mean((test$AIR - rf.pred)^2)
rf.rmse <- sqrt(rf.mse)
rf.mae  <- mean(abs(test$AIR - rf.pred))
cat("\n--- Random Forest ---\nMSE:", round(rf.mse,6), "| RMSE:", round(rf.rmse,6), "| MAE:", round(rf.mae,6), "\n")

plot(test$Date, test$AIR, type = "l", col = "black", lwd = 1.5,
     main = "Random Forest: Actual vs Predicted", xlab = "Date", ylab = "Weekly Return")
lines(test$Date, rf.pred, col = "darkgreen", lwd = 1.5, lty = 2)
abline(h = 0, col = "gray", lty = 3)
legend("topright", legend = c("Actual", "Random Forest"), col = c("black", "darkgreen"), lty = c(1, 2), lwd = 1.5)

plot(rf.pred, test$AIR, pch = 20, col = "darkgreen",
     main = "Random Forest: Scatter Plot", xlab = "Predicted", ylab = "Actual")
abline(0, 1, col = "red", lwd = 2)

# ---- 2.7 MODEL 6 - BOOSTING (GBM) ----
set.seed(1)
boost.fit <- gbm(AIR ~ BA + SAF + CAC + EURUSD + VIX + BRENT + US10YR + GE,
                 data = train, distribution = "gaussian",
                 n.trees = 5000, interaction.depth = 4, shrinkage = 0.01, cv.folds = 5)
best.trees <- gbm.perf(boost.fit, method = "cv")
cat("Optimal number of trees:", best.trees, "\n")
summary(boost.fit, n.trees = best.trees)

boost.pred <- predict(boost.fit, newdata = test, n.trees = best.trees)
boost.mse  <- mean((test$AIR - boost.pred)^2)
boost.rmse <- sqrt(boost.mse)
boost.mae  <- mean(abs(test$AIR - boost.pred))
cat("\n--- Boosting ---\nMSE:", round(boost.mse,6), "| RMSE:", round(boost.rmse,6), "| MAE:", round(boost.mae,6), "\n")

par(mfrow = c(1, 1))
plot(test$Date, test$AIR, type = "l", col = "black", lwd = 1.5,
     main = "Boosting: Actual vs Predicted", xlab = "Date", ylab = "Weekly Return")
lines(test$Date, boost.pred, col = "purple", lwd = 1.5, lty = 2)
abline(h = 0, col = "gray", lty = 3)
legend("topright", legend = c("Actual", "Boosting"), col = c("black", "purple"), lty = c(1, 2), lwd = 1.5)

plot(boost.pred, test$AIR, pch = 20, col = "purple",
     main = "Boosting: Scatter Plot", xlab = "Predicted", ylab = "Actual")
abline(0, 1, col = "red", lwd = 2)

# ============================================
# PART 3: SUMMARY TABLES
# ============================================

# ---- Table 2: Out-of-Sample MSE / RMSE / MAE ----
results <- data.frame(
  Model = c("OLS", "Lasso", "GAM", "Bagging", "Random Forest", "Boosting"),
  MSE   = round(c(ols.mse, lasso.mse, gam.mse, bag.mse, rf.mse, boost.mse), 6),
  RMSE  = round(c(ols.rmse, lasso.rmse, gam.rmse, bag.rmse, rf.rmse, boost.rmse), 6),
  MAE   = round(c(ols.mae, lasso.mae, gam.mae, bag.mae, rf.mae, boost.mae), 6)
)
print(results)
kbl(results, format = "html",
    caption = "Table 2: Out-of-Sample Forecasting Performance (Test Set)",
    align = "c") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"),
                full_width = FALSE, font_size = 14) %>%
  row_spec(0, bold = TRUE, background = "#2F5496", color = "white") %>%
  row_spec(4, bold = TRUE, background = "#D6E4F0")

# ---- Table 4: In-Sample vs Out-of-Sample R² and Adjusted R² ----
SS.tot       <- sum((test$AIR - mean(test$AIR))^2)
SS.tot.train <- sum((train$AIR - mean(train$AIR))^2)

r2.all <- c(
  1 - sum((test$AIR - ols.pred)^2) / SS.tot,
  1 - sum((test$AIR - lasso.pred)^2) / SS.tot,
  1 - sum((test$AIR - gam.pred)^2) / SS.tot,
  1 - sum((test$AIR - bag.pred)^2) / SS.tot,
  1 - sum((test$AIR - rf.pred)^2) / SS.tot,
  1 - sum((test$AIR - boost.pred)^2) / SS.tot
)
adj.r2.out <- 1 - (1 - r2.all) * (n.test - 1) / (n.test - p - 1)

r2.in.all <- c(
  summary(ols.fit)$r.squared,
  1 - sum((y.train - predict(lasso.mod, s = bestlam, newx = x.train))^2) / SS.tot.train,
  1 - sum((train$AIR - predict(gam.fit, newdata = train))^2) / SS.tot.train,
  1 - sum((train$AIR - predict(bag.fit, newdata = train))^2) / SS.tot.train,
  1 - sum((train$AIR - predict(rf.fit, newdata = train))^2) / SS.tot.train,
  1 - sum((train$AIR - predict(boost.fit, newdata = train, n.trees = best.trees))^2) / SS.tot.train
)
adj.r2.in <- 1 - (1 - r2.in.all) * (n.train - 1) / (n.train - p - 1)

r2.full <- data.frame(
  Model             = c("OLS", "Lasso", "GAM", "Bagging", "Random Forest", "Boosting"),
  R2_InSample       = round(r2.in.all, 4),
  AdjR2_InSample    = round(adj.r2.in, 4),
  R2_OutOfSample    = round(r2.all, 4),
  AdjR2_OutOfSample = round(adj.r2.out, 4)
)
print(r2.full)
kbl(r2.full, format = "html",
    caption = "Table 4: In-Sample vs Out-of-Sample R² and Adjusted R²",
    align = "c") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"),
                full_width = FALSE, font_size = 14) %>%
  row_spec(0, bold = TRUE, background = "#2F5496", color = "white") %>%
  row_spec(4, bold = TRUE, background = "#D6E4F0")

# ============================================
# PART 4: ROBUSTNESS CHECK
# ============================================

# ---- 4.1 OLS Diagnostic Tests ----
cat("\n========== OLS DIAGNOSTIC TESTS ==========\n")

cat("\n--- Shapiro-Wilk Test (Normality) ---\n")
print(shapiro.test(residuals(ols.fit)))

cat("\n--- Jarque-Bera Test (Normality) ---\n")
print(jarque.test(residuals(ols.fit)))

cat("\n--- Ljung-Box Test (Autocorrelation, lag=10) ---\n")
print(Box.test(residuals(ols.fit), lag = 10, type = "Ljung-Box"))

cat("\n--- Breusch-Pagan Test (Heteroscedasticity) ---\n")
print(bptest(ols.fit))

par(mfrow = c(1, 2))
plot(fitted(ols.fit), residuals(ols.fit),
     xlab = "Fitted Values", ylab = "Residuals",
     main = "Residuals vs Fitted", pch = 20, col = "gray30")
abline(h = 0, col = "red", lty = 2)
plot(train$Date, residuals(ols.fit), type = "l", col = "darkblue",
     xlab = "Date", ylab = "Residuals",
     main = "Residuals over Time")
abline(h = 0, col = "red", lty = 2)

# ---- 4.2 Alternative Split (70/30) ----
train2 <- data[data$Date < as.Date("2023-01-01"), ]
test2  <- data[data$Date >= as.Date("2023-01-01"), ]
n2 <- nrow(test2)
cat("\nRobustness - Train:", nrow(train2), "obs | Test:", n2, "obs\n")

# OLS
ols.fit2  <- lm(AIR ~ BA + SAF + CAC + EURUSD + VIX + BRENT + US10YR + GE, data = train2)
ols.pred2 <- predict(ols.fit2, newdata = test2)
ols.mse2  <- mean((test2$AIR - ols.pred2)^2)
ols.rmse2 <- sqrt(ols.mse2)
ols.mae2  <- mean(abs(test2$AIR - ols.pred2))

# Lasso
x.train2 <- as.matrix(train2[, c("BA","SAF","CAC","EURUSD","VIX","BRENT","US10YR","GE")])
y.train2 <- train2$AIR
x.test2  <- as.matrix(test2[, c("BA","SAF","CAC","EURUSD","VIX","BRENT","US10YR","GE")])
y.test2  <- test2$AIR
set.seed(1)
cv.lasso2   <- cv.glmnet(x.train2, y.train2, alpha = 1)
lasso.pred2 <- predict(cv.lasso2, s = cv.lasso2$lambda.min, newx = x.test2)
lasso.mse2  <- mean((y.test2 - lasso.pred2)^2)
lasso.rmse2 <- sqrt(lasso.mse2)
lasso.mae2  <- mean(abs(y.test2 - lasso.pred2))

# GAM
gam.fit2  <- gam(AIR ~ s(BA,4)+s(SAF,4)+s(CAC,4)+s(EURUSD,4)+s(VIX,4)+s(BRENT,4)+s(US10YR,4)+s(GE,4), data = train2)
gam.pred2 <- predict(gam.fit2, newdata = test2)
gam.mse2  <- mean((test2$AIR - gam.pred2)^2)
gam.rmse2 <- sqrt(gam.mse2)
gam.mae2  <- mean(abs(test2$AIR - gam.pred2))

# Bagging
set.seed(1)
bag.fit2  <- randomForest(AIR ~ BA + SAF + CAC + EURUSD + VIX + BRENT + US10YR + GE,
                          data = train2, mtry = 8, ntree = 500)
bag.pred2 <- predict(bag.fit2, newdata = test2)
bag.mse2  <- mean((test2$AIR - bag.pred2)^2)
bag.rmse2 <- sqrt(bag.mse2)
bag.mae2  <- mean(abs(test2$AIR - bag.pred2))

# Random Forest
set.seed(1)
rf.fit2  <- randomForest(AIR ~ BA + SAF + CAC + EURUSD + VIX + BRENT + US10YR + GE,
                         data = train2, mtry = 3, ntree = 500)
rf.pred2 <- predict(rf.fit2, newdata = test2)
rf.mse2  <- mean((test2$AIR - rf.pred2)^2)
rf.rmse2 <- sqrt(rf.mse2)
rf.mae2  <- mean(abs(test2$AIR - rf.pred2))

# Boosting
set.seed(1)
boost.fit2  <- gbm(AIR ~ BA + SAF + CAC + EURUSD + VIX + BRENT + US10YR + GE,
                   data = train2, distribution = "gaussian",
                   n.trees = 5000, interaction.depth = 4, shrinkage = 0.01, cv.folds = 5)
best.trees2 <- gbm.perf(boost.fit2, method = "cv", plot.it = FALSE)
boost.pred2 <- predict(boost.fit2, newdata = test2, n.trees = best.trees2)
boost.mse2  <- mean((test2$AIR - boost.pred2)^2)
boost.rmse2 <- sqrt(boost.mse2)
boost.mae2  <- mean(abs(test2$AIR - boost.pred2))

# ---- Table 3: Robustness MSE/RMSE/MAE ----
robust <- data.frame(
  Model      = c("OLS", "Lasso", "GAM", "Bagging", "Random Forest", "Boosting"),
  MSE_80_20  = round(c(ols.mse, lasso.mse, gam.mse, bag.mse, rf.mse, boost.mse), 6),
  MSE_70_30  = round(c(ols.mse2, lasso.mse2, gam.mse2, bag.mse2, rf.mse2, boost.mse2), 6),
  RMSE_80_20 = round(c(ols.rmse, lasso.rmse, gam.rmse, bag.rmse, rf.rmse, boost.rmse), 6),
  RMSE_70_30 = round(c(ols.rmse2, lasso.rmse2, gam.rmse2, bag.rmse2, rf.rmse2, boost.rmse2), 6),
  MAE_80_20  = round(c(ols.mae, lasso.mae, gam.mae, bag.mae, rf.mae, boost.mae), 6),
  MAE_70_30  = round(c(ols.mae2, lasso.mae2, gam.mae2, bag.mae2, rf.mae2, boost.mae2), 6)
)
print(robust)
kbl(robust, format = "html",
    caption = "Table 3: Robustness Check - 80/20 vs 70/30 Split",
    align = "c") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"),
                full_width = FALSE, font_size = 14) %>%
  row_spec(0, bold = TRUE, background = "#2F5496", color = "white") %>%
  row_spec(4, bold = TRUE, background = "#D6E4F0")

# ---- Table 5: Robustness R² ----
SS.tot2 <- sum((test2$AIR - mean(test2$AIR))^2)
r2.all2 <- c(
  1 - sum((test2$AIR - ols.pred2)^2) / SS.tot2,
  1 - sum((test2$AIR - lasso.pred2)^2) / SS.tot2,
  1 - sum((test2$AIR - gam.pred2)^2) / SS.tot2,
  1 - sum((test2$AIR - bag.pred2)^2) / SS.tot2,
  1 - sum((test2$AIR - rf.pred2)^2) / SS.tot2,
  1 - sum((test2$AIR - boost.pred2)^2) / SS.tot2
)
adj.r2.2 <- 1 - (1 - r2.all2) * (n2 - 1) / (n2 - p - 1)

r2.robust <- data.frame(
  Model       = c("OLS", "Lasso", "GAM", "Bagging", "Random Forest", "Boosting"),
  R2_80_20    = round(r2.all, 4),
  R2_70_30    = round(r2.all2, 4),
  AdjR2_80_20 = round(adj.r2.out, 4),
  AdjR2_70_30 = round(adj.r2.2, 4)
)
print(r2.robust)
kbl(r2.robust, format = "html",
    caption = "Table 5: Robustness Check - R² and Adjusted R² (80/20 vs 70/30)",
    align = "c") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"),
                full_width = FALSE, font_size = 14) %>%
  row_spec(0, bold = TRUE, background = "#2F5496", color = "white") %>%
  row_spec(4, bold = TRUE, background = "#D6E4F0")
b

