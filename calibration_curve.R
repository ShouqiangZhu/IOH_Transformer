# evaluation/calibration_curve.R
library(ggplot2)
library(scales)
library(Metrics)

# Load predictions
data <- read.csv("eval_predictions.csv")
data$pred_bin <- cut(data$pred_prob, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)
calib <- aggregate(data$label, by = list(bin = data$pred_bin), mean)
names(calib) <- c("Bin", "Observed")

# Expected (mean of predicted prob in each bin)
expected <- aggregate(data$pred_prob, by = list(bin = data$pred_bin), mean)
calib$Expected <- expected$x

# Plot calibration
ggplot(calib, aes(x = Expected, y = Observed)) +
  geom_line(color = "blue") +
  geom_abline(linetype = "dashed") +
  xlim(0,1) + ylim(0,1) +
  theme_minimal() +
  labs(title = "Calibration Curve", x = "Predicted", y = "Observed")

# ECE
ece <- sum(abs(calib$Observed - calib$Expected) * table(data$pred_bin) / nrow(data))
cat(sprintf("Expected Calibration Error (ECE): %.4f\n", ece))
