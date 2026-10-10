library(dplyr)
library(philentropy)
library(tibble)

# Run this script from the directory containing the CSV file, or replace this
# with the absolute path to your CSV file.
input_csv <- "V_Te1_struc_comp.csv"

# The background RDF is the mean of every structure except structures 22, 33 and 28.
target_structures <- c(22, 33, 87)

# Randomly select three structures from the background pool. Change this seed
# to obtain a different reproducible selection.
random_seed <- 20260920

# `check.names = FALSE` retains RDF columns such as v_2.50 exactly as written.
rdf_data <- read.csv(input_csv, check.names = FALSE)

# Identify the RDF columns and reconstruct their radial positions from names
# such as v_2.50, v_2.55, ..., v_9.95.
rdf_columns <- grep("^v_", names(rdf_data), value = TRUE)
radial_bins <- as.numeric(sub("^v_", "", rdf_columns))

if (!"structures" %in% names(rdf_data)) {
  stop("The CSV must contain a 'structures' column.")
}

if (!all(target_structures %in% rdf_data$structures)) {
  missing_targets <- setdiff(target_structures, rdf_data$structures)
  stop("Could not find structure(s): ", paste(missing_targets, collapse = ", "))
}

# Calculate the pointwise background mean only from structures other than 22, 33
# and 87. `na.rm = TRUE` safely handles any missing RDF bins.
background_data <- rdf_data %>%
  filter(!structures %in% target_structures) %>%
  select(all_of(rdf_columns))

if (nrow(background_data) == 0) {
  stop("No background structures remain after excluding 22 and 33 and 87.")
}

background_mean <- colMeans(background_data, na.rm = TRUE)

extract_structure_rdf <- function(structure_id) {
  structure_rdf <- rdf_data %>%
    filter(structures == structure_id) %>%
    select(all_of(rdf_columns))
  
  if (nrow(structure_rdf) != 1) {
    stop("Expected exactly one RDF row for structure ", structure_id,
         "; found ", nrow(structure_rdf), ".")
  }
  
  as.numeric(unlist(structure_rdf, use.names = FALSE))
}

g_22 <- extract_structure_rdf(22)
g_33 <- extract_structure_rdf(33)
g_87 <- extract_structure_rdf(87)
g_background <- as.numeric(background_mean[rdf_columns])

# Returns the Jensen-Shannon distance: sqrt(Jensen-Shannon divergence).
# `est.prob = "empirical"` converts the RDF values in each radial window into
# probability distributions before calculating the divergence.
get_jsd_by_window <- function(ref_vec, cmp_vec, r_vec, r_low, r_high, label) {
  idx <- which(r_vec >= r_low & r_vec < r_high)
  
  if (length(idx) == 0) {
    stop("No RDF bins found in window ", label)
  }
  
  x <- rbind(ref_vec[idx], cmp_vec[idx])
  
  if (any(!is.finite(x))) {
    stop("Non-finite RDF values found in window ", label)
  }
  
  jsd_divergence <- as.numeric(JSD(x, unit = "log2", est.prob = "empirical"))
  
  tibble(
    radial_regime = label,
    jsd_distance = sqrt(jsd_divergence)
  )
}

radial_windows <- tribble(
  ~r_low, ~r_high, ~label,
  0,      4,       "0–4 Å",
  4,      7,       "4–7 Å",
  7,      10,      "7–10 Å"
)

calculate_windowed_jsd <- function(ref_vec, cmp_vec) {
  bind_rows(lapply(seq_len(nrow(radial_windows)), function(i) {
    get_jsd_by_window(
      ref_vec,
      cmp_vec,
      radial_bins,
      radial_windows$r_low[i],
      radial_windows$r_high[i],
      radial_windows$label[i]
    )
  }))
}

# Calculates a leave-one-out background mean for one randomly selected
# background structure. Thus its own RDF, and the RDFs of structures 16 and 28,
# are excluded from the mean used for its comparison.
get_leave_one_out_background <- function(structure_id) {
  remaining_background <- rdf_data %>%
    filter(!structures %in% c(target_structures, structure_id)) %>%
    select(all_of(rdf_columns))
  
  if (nrow(remaining_background) == 0) {
    stop("No structures remain to construct the leave-one-out background for structure ",
         structure_id, ".")
  }
  
  as.numeric(colMeans(remaining_background, na.rm = TRUE)[rdf_columns])
}

# 1. Individual RDF of structure 22 versus individual RDF of structure 33.
jsd_22_vs_33 <- calculate_windowed_jsd(g_22, g_33)

# 2. Individual RDF of structure 22 versus individual RDF of structure 87.
jsd_22_vs_87 <- calculate_windowed_jsd(g_22, g_87)

# 3. Individual RDF of structure 33 versus individual RDF of structure 87.
jsd_33_vs_87 <- calculate_windowed_jsd(g_33, g_87)

# 4. Individual RDF of structure 16 versus the mean RDF of all background structures.
jsd_22_vs_background <- calculate_windowed_jsd(g_22, g_background)

# 5. Individual RDF of structure 28 versus the mean RDF of all background structures.
jsd_33_vs_background <- calculate_windowed_jsd(g_33, g_background)

# 6. Individual RDF of structure 87 versus the mean RDF of all background structures.
jsd_87_vs_background <- calculate_windowed_jsd(g_87, g_background)

# 4. Randomly choose three background structures. For each one, compare its RDF
# against a mean RDF that excludes that structure as well as structures 16 and 28.
background_structure_ids <- setdiff(rdf_data$structures, target_structures)

if (length(background_structure_ids) < 3) {
  stop("At least three background structures are required for random sampling.")
}

set.seed(random_seed)
random_structure_ids <- sample(background_structure_ids, size = 10, replace = FALSE)

jsd_random_vs_leave_one_out_background <- bind_rows(lapply(random_structure_ids, function(structure_id) {
  individual_rdf <- extract_structure_rdf(structure_id)
  leave_one_out_background <- get_leave_one_out_background(structure_id)
  
  calculate_windowed_jsd(individual_rdf, leave_one_out_background) %>%
    mutate(structure = structure_id, .before = radial_regime)
}))

cat("\nJSD distance: Structure 22 vs Structure 33\n")
print(jsd_22_vs_33)

cat("\nJSD distance: Structure 22 vs Structure 87\n")
print(jsd_22_vs_87)

cat("\nJSD distance: Structure 33 vs Structure 87\n")
print(jsd_33_vs_87)

cat("\nJSD distance: Structure 22 vs background mean (all structures except 22 and 33 amd 87)\n")
print(jsd_22_vs_background)

cat("\nJSD distance: Structure 33 vs background mean (all structures except 22 and 33 amd 87)\n")
print(jsd_33_vs_background)

cat("\nJSD distance: Structure 33 vs background mean (all structures except 22 and 33 amd 87)\n")
print(jsd_87_vs_background)

cat("\nRandomly selected background structures (seed = ", random_seed, "):\n", sep = "")
print(random_structure_ids)

cat("\nJSD distance: Each random structure vs its leave-one-out background mean\n")
print(n=60, jsd_random_vs_leave_one_out_background)
