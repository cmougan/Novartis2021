library(tidyr)
library(dplyr)
library(ggplot2)
theme_set(theme_minimal())

submissions <- read.csv('data/submissions_data.csv')

submissions

submissions %>% 
  pivot_wider(names_from=setting, values_from=c(accuracy, interval)) %>% 
  ggplot(aes(x = accuracy_valid, y = accuracy_lb, color = model)) + 
  geom_point() +
  geom_label(aes(label=submission))



submissions %>% 
  pivot_wider(names_from=setting, values_from=c(accuracy, interval)) %>% 
  ggplot(aes(x = interval_valid, y = interval_lb, color = model)) + 
  geom_point() + 
  geom_label(aes(label=submission))
