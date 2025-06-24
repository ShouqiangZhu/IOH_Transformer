# evaluation/stat_analysis.R
library(dplyr)
library(broom)

df <- read.csv("clinical_outcomes.csv")

# Logistic regression for AKI
model_aki <- glm(AKI ~ hypotension_burden + age + ASA + surgery_type, data = df, family = "binomial")
summary(model_aki)
exp(confint(model_aki))

# Logistic regression for AKD
model_akd <- glm(AKD ~ hypotension_burden + age + ASA + surgery_type, data = df, family = "binomial")
summary(model_akd)
exp(confint(model_akd))
