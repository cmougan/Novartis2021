# Clean environment
rm(list = ls()); gc()

# Import libraries
library(data.table)

# ######################

# +
# Prediction intervals metric
alpha=0.2

msis <- function(board, market_size) {
  wide_int = abs(board$upper - board$lower)
  outside = (2/alpha * (board$lower - board$actuals) *
                                      (board$actuals < board$lower) + # If it's outside, it adds error
                             2/alpha * (board$actuals - board$upper) *
                                      (board$actuals > board$upper)
                             )      
  board$uncertainty <- wide_int + outside
  
  #calculates \Delta_b^r
  metric = board[, .(mean(uncertainty)), by=c("region")]  
  
  return (mean(metric$V1 / market_size$V1))

}

# +
ComputeByBrand <- function(dt, market_size) {
  # Get actuals and prediction
  actuals <- dt$actuals
  predictions <- dt$predictions
  lower <- dt$lower
  upper <- dt$upper

  #Sanity checks
  actuals <- actuals[!is.na(actuals)]
  predictions <- predictions[!is.na(predictions)]
  if (length(actuals) != length(predictions))
    stop("Predictions and actuals should have the same length and have no NAs. Please, check your solution.")
    
  #calculates MAE_c^b
  mae_country <- dt[, .(abs(sum(predictions) - sum(actuals))), by = c("month")]
  mae_country <- mean(mae_country$V1) / sum(market_size$V1)
    
  #calculates MAE_r^b
  dt$abs_error <- abs(predictions - actuals)
  mae_region <- dt[, .(abs_error), by = c("region", "month")]
  mae_region <- mae_region[, .(mean(abs_error)), by=c("region")]
    
  mae_region <- mean(mae_region$V1 / market_size$V1)
  
  return (mae_country + mae_region)
    
}
# -

ComputeMetrics <- function(solution, sales_train, ground_truth) {
  
  ground_truth <- ground_truth[, .(month, region, brand, actuals = sales)]
  solution <- solution[, .(month, region, brand, predictions = sales, upper = upper, lower = lower)]

  board <- merge(solution, ground_truth, by = c("brand", "region", "month"), all.x = T)
  board <- board[, ID := paste(brand, region, month, sep = '; ')]
  board <- board[order(ID)]
  board$month <- as.Date(paste(board$month,"-01",sep=""))

  brand_1_board <- board[brand == "brand_1"]
  brand_2_board <- board[brand == "brand_2"]
    
  #the average brand_12_market - terms <m_c>, <m_r> - are calculated here
  #only regions in the ground_truth set are selected    
  market_size <- sales_train[region %in% unique(solution$region) & brand == "brand_12_market", .(mean(sales)), by='region']  
    
  # Compute MAE metric
  brand_1_metric <- ComputeByBrand(brand_1_board, market_size)
  brand_2_metric <- ComputeByBrand(brand_2_board, market_size)
  metric <- 10000*(brand_1_metric + brand_2_metric) / 2

  # Compute interval metric
  brand_1_interval_metric <- msis(brand_1_board, market_size)
  brand_2_interval_metric <- msis(brand_2_board, market_size)
  interval_metric <- 10000*(brand_1_interval_metric + brand_2_interval_metric) / 2
    
  return(list(metric, interval_metric))
}
