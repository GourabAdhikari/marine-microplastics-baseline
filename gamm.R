# Usage: Rscript gamm.R train.csv test.csv pred.csv
# Fits the GAMM on train.csv and writes predictions for test.csv.
library(mgcv)

args <- commandArgs(trailingOnly = TRUE)
train <- read.csv(args[1])
test <- read.csv(args[2])
train$study <- factor(train$study)

families <- Filter(function(f) length(unique(train[[f]])) > 1, c("fam_pump", "fam_bulk"))
formula <- reformulate(c(
  's(lat, lon, bs = "sos", k = 100)', "s(year, k = 8)", "month_sin", "month_cos",
  "log_mesh", "log_depth", families, 's(study, bs = "re")'
), response = "y")
fit <- bam(formula, data = train, method = "fREML", discrete = TRUE)

seen <- test$study %in% levels(train$study)
test$study <- factor(ifelse(seen, test$study, levels(train$study)[1]), levels = levels(train$study))
pred <- predict(fit, test, exclude = 's(study)')
pred[seen] <- predict(fit, test[seen, ])
write.csv(data.frame(pred = pred), args[3], row.names = FALSE)
