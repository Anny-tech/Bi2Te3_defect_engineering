library(readxl)
library(dplyr)
library(philentropy)
library(tibble)

# install.packages('philentropy', dependencies = TRUE)

# change working directory
setwd("/home/digifort/Pictures/paper_1_figs/upload")

# read Structure 103 + radial bins
jsd_input <- read_xlsx("jsd_input.xlsx", sheet = 1)

radial_bins <- jsd_input[["radial bins"]]
g_103       <- jsd_input[["Structure 103"]]

cl1_mean <- jsd_input[["Cluster_1_mean"]]
cl4_mean <- jsd_input[["Cluster_4_mean"]]
cl5_mean <- jsd_input[["Cluster_5_mean"]]
cl7_mean <- jsd_input[["Cluster_7_mean"]]

rdf <- tibble(
  r          = radial_bins,
  struct_103 = g_103,
  cl1_mean   = cl1_mean,
  cl4_mean   = cl4_mean,
  cl5_mean   = cl5_mean,
  cl7_mean   = cl7_mean,
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

# cluster 1
jsd_cl1 <- bind_rows(
  get_jsd_by_window(rdf$struct_103, rdf$cl1_mean, rdf$r, 2.5, 4.0, "2.5–4 Å"),
  get_jsd_by_window(rdf$struct_103, rdf$cl1_mean, rdf$r, 4.0, 7.0, "4–7 Å"),
  get_jsd_by_window(rdf$struct_103, rdf$cl1_mean, rdf$r, 7.0, 10.0, "7–10 Å"),
  get_jsd_by_window(rdf$struct_103, rdf$cl1_mean, rdf$r, 2.5, 10.0, "Full range")
) %>%
  mutate(cluster = 1)

# cluster 4
jsd_cl4 <- bind_rows(
  get_jsd_by_window(rdf$struct_103, rdf$cl4_mean, rdf$r, 2.5, 4.0, "2.5–4 Å"),
  get_jsd_by_window(rdf$struct_103, rdf$cl4_mean, rdf$r, 4.0, 7.0, "4–7 Å"),
  get_jsd_by_window(rdf$struct_103, rdf$cl4_mean, rdf$r, 7.0, 10.0, "7–10 Å"),
  get_jsd_by_window(rdf$struct_103, rdf$cl4_mean, rdf$r, 2.5, 10.0, "Full range")
) %>%
  mutate(cluster = 4)

# cluster 5
jsd_cl5 <- bind_rows(
  get_jsd_by_window(rdf$struct_103, rdf$cl5_mean, rdf$r, 2.5, 4.0, "2.5–4 Å"),
  get_jsd_by_window(rdf$struct_103, rdf$cl5_mean, rdf$r, 4.0, 7.0, "4–7 Å"),
  get_jsd_by_window(rdf$struct_103, rdf$cl5_mean, rdf$r, 7.0, 10.0, "7–10 Å"),
  get_jsd_by_window(rdf$struct_103, rdf$cl5_mean, rdf$r, 2.5, 10.0, "Full range")
) %>%
  mutate(cluster = 5)

# cluster 7
jsd_cl7 <- bind_rows(
  get_jsd_by_window(rdf$struct_103, rdf$cl7_mean, rdf$r, 2.5, 4.0, "2.5–4 Å"),
  get_jsd_by_window(rdf$struct_103, rdf$cl7_mean, rdf$r, 4.0, 7.0, "4–7 Å"),
  get_jsd_by_window(rdf$struct_103, rdf$cl7_mean, rdf$r, 7.0, 10.0, "7–10 Å"),
  get_jsd_by_window(rdf$struct_103, rdf$cl7_mean, rdf$r, 2.5, 10.0, "Full range")
) %>%
  mutate(cluster = 7)

jsd_results <- bind_rows(jsd_cl1, jsd_cl4, jsd_cl5, jsd_cl7) %>%
  select(cluster, window, jsd)


jsd_results
