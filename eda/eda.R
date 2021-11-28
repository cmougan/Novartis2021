library(ggplot2)
library(dplyr)
library(patchwork)

theme_set(theme_minimal())

train <- read.csv("eda/train_features.csv")

train %>% 
  mutate(
    brand = if_else(brand == 'brand_1', 'Brand 1', 'Brand 2')
  ) %>% 
  ggplot((aes(x = coalesce(inverse_tier_f2f, 0), y = target, color=brand))) +
  geom_smooth() + 
  xlab('F2F interactions') +
  ylab('Sales') +
  labs(color='') + 
  ggtitle("Brand 1 sales are more impacted by F2F meetings") +
  ggsave('high_tier_feats.png', scale=0.5)


train %>% 
  ggplot((aes(x = coalesce(hcp_distinct_Internal.medicine...pneumology, 0), y = target, color=brand))) +
  # geom_point(alpha = 0.2) +
  geom_smooth() +
  xlab('Pneumologist meetings') +
  ylab('Sales') +
  labs(color='Brand')


train %>% 
  ggplot((aes(x = coalesce(inverse_tier_f2f, 0), y = target, color=brand))) +
  # geom_point(alpha = 0.2) + 
  geom_smooth() + 
  xlab('High tier f2f meetings') +
  ylab('Sales') +
  labs(color='Brand')



train %>% 
  ggplot((aes(x = coalesce(inverse_tier_f2f, 0), y = target, color=brand))) +
  # geom_point(alpha = 0.2) + 
  geom_smooth() + 
  xlab('High tier f2f meetings') +
  ylab('Sales') +
  labs(color='Brand')



train %>% 
  ggplot((aes(x = coalesce(inverse_tier_f2f, 0), y = target, color=brand))) +
  # geom_point(alpha = 0.2) + 
  geom_smooth() + 
  xlab('High tier f2f meetings') +
  ylab('Sales') +
  labs(color='Brand')



