# =====================================================================
# 0. Libraries
# =====================================================================
library(data.table)
library(dplyr)
library(ranger)
library(terra)
library(pROC)
library(tidyr)

set.seed(1234)

# =====================================================================
# 1. Load data
# =====================================================================

TRAIN_CSV <- "/mnt/eo/EO4Backcasting/_intermediates/training_data_topo.csv"
df <- fread(TRAIN_CSV)

# Only keep rows with valid BAP
df <- df[df$bap_available == TRUE]

# =====================================================================
# 2. Create target variable (binary)
# =====================================================================

df <- df %>%
  mutate(
    target = case_when(
      state == "undisturbed"           ~ 0L,
      state == "disturbed" & ysd <= 15 ~ 1L,
      state == "disturbed" & ysd > 15  ~ 0L,
      TRUE ~ NA_integer_
    )
  )

df <- df %>% filter(!is.na(target))

pred_cols <- c("b1","b2","b3","b4","b5","b6")

df <- df %>% drop_na(all_of(pred_cols))

cat("Training rows:", nrow(df), "\n")
print(table(df$target))


# =====================================================================
# 3. Train/validation split
# =====================================================================

set.seed(1234)
idx <- sample(seq_len(nrow(df)))
n_train <- floor(0.8 * nrow(df))

train_df <- df[idx[1:n_train], ]
val_df   <- df[idx[(n_train+1):nrow(df)], ]


# =====================================================================
# 4. Ranger RF model
# =====================================================================

rf <- ranger(
  formula = as.factor(target) ~ b1 + b2 + b3 + b4 + b5 + b6,
  data    = train_df,
  num.trees   = 500,
  mtry        = 3,
  probability = TRUE,
  importance  = "impurity"
)

print(rf)


# =====================================================================
# 5. Validation
# =====================================================================

# probability of class "1" = recent disturbance
val_prob <- predict(rf, val_df)$predictions[,2]

# class threshold 0.5
val_pred <- ifelse(val_prob > 0.5, 1, 0)

acc <- mean(val_pred == val_df$target)
cat("\nValidation accuracy:", acc, "\n")

cat("\nConfusion matrix:\n")
print(table(pred=val_pred, true=val_df$target))

roc_obj <- roc(val_df$target, val_prob)
cat("\nAUC:", auc(roc_obj), "\n")


# =====================================================================
# 6. Save model
# =====================================================================

saveRDS(rf, "/mnt/eo/EO4Backcasting/_models/rf_bap_binary.rds")
saveRDS(list(pred_cols = pred_cols),
        "/mnt/eo/EO4Backcasting/_models/rf_bap_recentdist_binary_meta.rds")

cat("\nModel + metadata saved.\n")
