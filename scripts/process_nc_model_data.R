# process data from NC forecasting collab repo to respilens

library(tidyverse)
library(readr)
library(dplyr)
library(httr)

#setwd
setwd("../NC_Forecasting_Collab")


# function for converting from sample trajectories to quantile representation or cdf
convert_from_sample <- function(grouped_model_out_tbl, new_output_type,
                                new_output_type_id) {
  if (new_output_type == "mean") {
    model_out_tbl_transform <- grouped_model_out_tbl %>%
      dplyr::reframe(
        value = mean(.data[["value"]]),
        output_type_id = new_output_type_id
      )
  } else if (new_output_type == "median") {
    model_out_tbl_transform <- grouped_model_out_tbl %>%
      dplyr::reframe(
        value = mean(.data[["value"]]),
        output_type_id = new_output_type_id
      )
  } else if (new_output_type == "quantile") {
    model_out_tbl_transform <- grouped_model_out_tbl %>%
      dplyr::reframe(
        value = stats::quantile(.data[["value"]], as.numeric(new_output_type_id), names = FALSE),
        output_type_id = new_output_type_id
      )
  } else if (new_output_type == "cdf") {
    model_out_tbl_transform <- grouped_model_out_tbl %>%
      dplyr::reframe(
        value = stats::ecdf(.data[["value"]])(as.numeric(new_output_type_id)),
        output_type_id = new_output_type_id
      )
  }
  # update output_type and output_type_id columns
  model_out_tbl_transform <- model_out_tbl_transform %>%
    dplyr::mutate(output_type = new_output_type) 
  return(model_out_tbl_transform)
}

# function for reading in model outputs, formatting and combining based on format of current folders etc
read_forecasts <- function(file, pathogen = "flu", target = "hosp"){
  
  # # read in data, save filenames as column
  # modeloutput <-
  #   lapply(seq_along(modeloutput_files), function(i) {
  #     df <- read_csv(modeloutput_files[i], col_types = cols(location = col_character()))
  #     df <- df %>% mutate(model_id = basename(dirname(modeloutput_files[i])))  # Add filename as model_id
  #     return(df)
  #   }) %>%
  #   bind_rows() %>%
  #   mutate(model_id = gsub(paste0(proj_date,"-"),"",gsub(".csv", "", model_id)))
  # 
  modeloutput <- read_csv(file, col_types = cols(location = col_character())) %>%
    mutate(model_id = basename(dirname(file)))
  
  # filter model_output to 37000, 37 
  locs <- c("37000", "37", paste("NC",pathogen,target, sep = "_"))
  modeloutput <- modeloutput %>%
    filter(location %in% locs) %>%
    # select other than if model_id == influpaint and location == 37
    filter(!(model_id == "UNC_IDD-InfluPaint" & location == "37")) %>%
    mutate(location = "37")
  
  # if modeloutput for a given model_id has only 'sample' in output_type, create the quantiles
  modeloutput_2 <- lapply(unique(modeloutput$model_id), function(i) {
    df <- modeloutput %>% filter(model_id == i)
    if (n_distinct(df$output_type) == 1 & unique(df$output_type) == "samples") {
      quantiles <-  c(0.01,0.025, seq(0.050, 0.9, 0.05), 0.95, 0.975,0.99)
      return(convert_from_sample(df %>% group_by(horizon, target_end_date, location, target, reference_date, model_id),"quantile",quantiles))
    } else {
      return(df)
    }
  }) %>%
    bind_rows()
  
  return(modeloutput_2)
  
}

write_csv_with_dir <- function(data, file_path) {
  # Extract the directory part from the file path
  dir_path <- dirname(file_path)
  
  # Check if the directory exists, and create it if not
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
  }
  
  # Write the CSV file
  write_csv(data, file_path)
}

# options ---------
pathogen <- "flu"
target <- "hosp"

# read files from flu_modeloutput ----------------------------------------------

# get csv files in flu_modeloutput

output_files <- list.files("flu_modeloutput", recursive = TRUE, 
                           full.names = TRUE,
                           pattern = "^\\d{4}-\\d{2}-\\d{2}.*\\.csv$")
#remove files with the string "/JHU_IDD-hierarchSIM/"
output_files <- output_files[!grepl("/JHU_IDD-hierarchSIM/", output_files)]
output_files <- output_files[!grepl("2025-03-22-UNC_IDD", output_files)]

# loop through output_files and save them in respilens folder
for (file in output_files){
  modeloutput <- read_forecasts(file, pathogen = "flu", target = "hosp")
  new_filename <- file.path("../RespiLens_fork/Flusight-forecast-hub/model-output", 
                            unique(modeloutput$model_id),
                            paste0(unique(modeloutput$reference_date),
                                   "_",unique(modeloutput$model_id),".csv"))
  write_csv_with_dir(modeloutput,new_filename)
}

# extract forecast date from flu_modeloutput files

forecast_dates <- unique(str_extract(output_files, "\\d{4}-\\d{2}-\\d{2}"))


# process target data ---------

# list all files in target_data folder, and get the latest date

target_files <- list.files("nc_data", recursive = TRUE, 
                           full.names = TRUE)
target_files <- target_files[sapply(strsplit(target_files, "/"), length) == 2]

# 1. Filter for hosp_admissions files
str_grab <- ifelse(target == "hosp","hosp_admissions","ED_visits_total")

target_files <- target_files[grepl(paste0(str_grab,"\\.csv$"), target_files)]

# 2. Extract dates
dates <- gsub(paste0("nc_data/(\\d{8})_",str_grab,"\\.csv$"), "\\1", target_files)

# 3. Find the latest date
target_data_latest_date <- max(dates)

# 4. Construct the file path for the latest hosp_admissions file
target_data_file <- paste0("nc_data/", target_data_latest_date, "_",str_grab,".csv")

# 5. Verify and print the result
if (target_data_file %in% target_files) {
  print(target_data_file)
} else {
  print("Latest hosp_admissions file not found.")
}

loc_dic <- read_csv("../RespiLens_fork/Flusight-forecast-hub/auxiliary-data/locations.csv")

# Read in NC DPH data (the ground truth)
target_data <- read_csv(target_data_file) %>%
  mutate(date = lubridate::as_date(end_week_date)) %>%
  # mutate(date = lubridate::mdy(end_week_date)) %>%
  filter(date >= "2023-09-01") %>%
  rename(value = paste("flu",target,sep = "_")) %>%
  mutate(location = "37",
         abbreviation = "NC",
         location_name = "North Carolina") %>%
  left_join(loc_dic %>% mutate(location = as.character(location)), by = c("abbreviation","location","location_name")) %>%
  mutate(weekly_rate = value/population) %>%
  select(date, location, abbreviation, location_name, value, weekly_rate)

# date,location,location_name,value,weekly_rate
write_csv_with_dir(target_data,"../RespiLens_fork/Flusight-forecast-hub/target-data/target-hospital-admissions.csv")


# read in flu ensemble from raw github ---------

lapply(forecast_dates, function(i) {
  url <- paste0("https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/refs/heads/main/model-output/FluSight-ensemble/", i, "-FluSight-ensemble.csv")
  
  # Check if the URL exists
  response <- HEAD(url) # HEAD request is faster than GET for checking existence
  if (response$status_code == 200) {
    tryCatch({ #use tryCatch to handle potential errors during read_csv.
      flu_ensemble <- read_csv(url) %>%
        filter(location == "37") %>%
        mutate(model_id = "Flusight-ensemble")
      
      new_filename <- file.path("../RespiLens_fork/Flusight-forecast-hub/model-output",
                                unique(flu_ensemble$model_id),
                                paste0(unique(flu_ensemble$reference_date),
                                       "_", unique(flu_ensemble$model_id), ".csv"))
      
      write_csv_with_dir(flu_ensemble, new_filename)
    }, error = function(e) {
      warning(paste("Error processing", url, ":", e$message))
    })
    
  } else {
    warning(paste("URL does not exist:", url))
  }
})

# Create ensembles ---------

extract_csv_files_by_date <- function(file_paths, reference_date) {
  # Construct the pattern to match the date and start of filename
  date_pattern <- paste0(reference_date, "-")
  # Use grep to find file paths containing the reference date and starting with the date
  matching_files <- grep(paste0("^", date_pattern, ".*\\.csv$"), basename(file_paths), value = TRUE)
  #return the full file path.
  return(file_paths[basename(file_paths) %in% matching_files])
}

# loop through each available date
lapply(forecast_dates, function(i) {

    # Read in the model outputs for the given date
  model_outputs_files <- extract_csv_files_by_date(list.files("flu_modeloutput", recursive = TRUE, 
                                    full.names = TRUE),i)

  # Read in the model outputs
  model_outputs <- lapply(model_outputs_files, read_forecasts)
  
  # Combine the model outputs
  combined_model_outputs <- bind_rows(model_outputs) %>%
    # round output_type_id
    mutate(output_type_id = round(output_type_id, 3)) 
  
  # filter to the same horizon
  combined_model_outputs <- combined_model_outputs %>%
    filter(horizon %in% 0:3)
  
  mean_ens <- combined_model_outputs |>
    hubEnsembles::simple_ensemble(
      model_id = "simple-ensemble-mean"
    )
  
  # Write the ensemble to a CSV file
  new_filename <- file.path("../RespiLens_fork/Flusight-forecast-hub/model-output",
                            unique(mean_ens$model_id),
                            paste0(i, "_", unique(mean_ens$model_id), ".csv"))
  
  write_csv_with_dir(mean_ens, new_filename)
})
