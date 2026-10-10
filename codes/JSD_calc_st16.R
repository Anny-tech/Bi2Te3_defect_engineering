library(readxl)
library(dplyr)
library(philentropy)
library(tibble)

# install.packages('philentropy', dependencies = TRUE)

# change working directory
setwd("/home/digifort/Pictures/paper_1_figs/upload")

# read Structure 16 + radial bins
jsd_input <- read_xlsx("jsd_input_st16.xlsx", sheet = 1)

radial_bins <- jsd_input[["radial bins"]]
g_16      <- jsd_input[["Structure 16"]]

cl3_mean <- jsd_input[["Cluster_3_mean"]]
cl6_mean <- jsd_input[["Cluster_6_mean"]]
#cl5_mean <- jsd_input[["Cluster_5_mean"]]
cl4_mean <- jsd_input[["Cluster_4_mean"]]

rdf <- tibble(
  r          = radial_bins,
  struct_16 = g_16,
  cl3_mean   = cl3_mean,
  cl6_mean   = cl6_mean,
  cl4_mean   = cl4_mean,
)

get_jsd_by_window <- function(ref_vec, cmp_vec, r_vec, r_low, r_high, label) {
  idx <- which(r_vec >= r_low & r_vec < r_high)
  
  x <- rbind(
    ref_vec[idx],
    cmp_vec[idx]
  )
  
  jsd_div = as.numeric(JSD(x, unit = "log2", est.prob = "empirical"))
  
  tibble(
    window = label,
    jsd = sqrt(jsd_div)
  )
}

# cluster 3
jsd_cl3 <- bind_rows(
  get_jsd_by_window(rdf$struct_16, rdf$cl3_mean, rdf$r, 2.5, 4.0, "2.5–4 Å"),
  get_jsd_by_window(rdf$struct_16, rdf$cl3_mean, rdf$r, 4.0, 7.0, "4–7 Å"),
  get_jsd_by_window(rdf$struct_16, rdf$cl3_mean, rdf$r, 7.0, 10.0, "7–10 Å"),
  get_jsd_by_window(rdf$struct_16, rdf$cl3_mean, rdf$r, 2.5, 10.0, "Full range")
) %>%
  mutate(cluster = 3)

# cluster 6
jsd_cl6 <- bind_rows(
  get_jsd_by_window(rdf$struct_16, rdf$cl6_mean, rdf$r, 2.5, 4.0, "2.5–4 Å"),
  get_jsd_by_window(rdf$struct_16, rdf$cl6_mean, rdf$r, 4.0, 7.0, "4–7 Å"),
  get_jsd_by_window(rdf$struct_16, rdf$cl6_mean, rdf$r, 7.0, 10.0, "7–10 Å"),
  get_jsd_by_window(rdf$struct_16, rdf$cl6_mean, rdf$r, 2.5, 10.0, "Full range")
) %>%
  mutate(cluster = 6)

# cluster 4
jsd_cl4 <- bind_rows(
  get_jsd_by_window(rdf$struct_16, rdf$cl4_mean, rdf$r, 2.5, 4.0, "2.5–4 Å"),
  get_jsd_by_window(rdf$struct_16, rdf$cl4_mean, rdf$r, 4.0, 7.0, "4–7 Å"),
  get_jsd_by_window(rdf$struct_16, rdf$cl4_mean, rdf$r, 7.0, 10.0, "7–10 Å"),
  get_jsd_by_window(rdf$struct_16, rdf$cl4_mean, rdf$r, 2.5, 10.0, "Full range")
) %>%
  mutate(cluster = 4)

# cluster 7
#jsd_cl7 <- bind_rows(
#  get_jsd_by_window(rdf$struct_16, rdf$cl7_mean, rdf$r, 2.5, 4.0, "2.5–4 Å"),
#  get_jsd_by_window(rdf$struct_16, rdf$cl7_mean, rdf$r, 4.0, 7.0, "4–7 Å"),
#  get_jsd_by_window(rdf$struct_16, rdf$cl7_mean, rdf$r, 7.0, 10.0, "7–10 Å"),
#  get_jsd_by_window(rdf$struct_16, rdf$cl7_mean, rdf$r, 2.5, 10.0, "Full range")
#) %>%
#  mutate(cluster = 7)

jsd_results <- bind_rows(jsd_cl3, jsd_cl6, jsd_cl4) %>%
  select(cluster, window, jsd)


jsd_results