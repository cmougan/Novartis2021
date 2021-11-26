library(ggplot2)
library(dplyr)

theme_set(theme_minimal())

val_gbm <- read.csv('data/validation/empty_extractor_target_encoder_val.csv') %>% rename(pred = sales)
val_lm <- read.csv('data/validation/linear_model_simple_val.csv') %>% rename(pred = sales)
df <- read.csv('data/data_raw/sales_train.csv')

valid_df <- df %>% inner_join(val_gbm, by=c("month", "region", "brand"))
valid_df$month <- as.Date(paste(valid_df$month, "01", sep = '-'))
valid_df_lm <- df %>% inner_join(val_lm, by=c("month", "region", "brand"))
valid_df_lm$month <- as.Date(paste(valid_df_lm$month, "01", sep = '-'))

valid_df %>% 
  # filter(region == sample(valid_df$region %>% unique(), 1)) %>% 
  filter(region == "region_6") %>% 
  ggplot() +
  geom_line(aes(x = month, y = pred, color = brand, group = brand), linetype = 'dotted') +
  geom_line(aes(x = month, y = sales, color = brand, group = brand), size = 1) + 
  geom_linerange(aes(x = month, ymin = upper, ymax = lower, color = brand))



valid_df_lm %>% 
  # filter(region == sample(valid_df$region %>% unique(), 1)) %>% 
  filter(region == "region_6") %>% 
  ggplot() +
  geom_line(aes(x = month, y = pred, color = brand, group = brand), linetype = 'dotted') +
  geom_line(aes(x = month, y = sales, color = brand, group = brand), size = 1) + 
  geom_linerange(aes(x = month, ymin = upper, ymax = lower, color = brand))
