library(evd)
#library(evir)

data <- read.csv('/Users/fabienaugsburger/Documents/GitHub/master-project-cleaned/data/climatology/daily_winter_season/wind_speed_cluster_3_mostly_winter.csv')
data<- data$X0
data_subset<- data[0:30000]

# try a fit 

fit <- fgev(data_subset)
print(fit)
# Parameters from the fit
loc <- fit$estimate["loc"]
scale <- fit$estimate["scale"]
shape <- fit$estimate["shape"]

# Define a specific value for which to calculate the return period
value <- 38.23

# Calculate the return period
return_period <- 1 / (1 - pgev(value, loc = loc, scale = scale, shape = shape))
cat("Return period for value", value, "is approximately:", return_period, "years\n")

# Plot the density
hist(data, prob = TRUE, main = "GEV Density with Return Period Highlighted", xlab = "Value", xlim = c(min(data), max(data)))
curve(dgev(x, loc = loc, scale = scale, shape = shape), add = TRUE, col = "blue")

# Add a vertical line for the value
abline(v = value, col = "red", lwd = 2, lty = 2)5
text(value, 0.05, paste0("T ≈ ", round(return_period, 1), " years"), col = "red", pos = 4)
     
