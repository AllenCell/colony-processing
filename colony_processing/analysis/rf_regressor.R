library(randomForest)

# Read data
data = read.csv('\\\\allen\\aics\\microscopy\\Data\\fov_in_colony_rnd\\data_analysis_fov.csv')


# Add protocol and piezo info 
data$protocol <- ifelse(data$structure_name %in% c("SMC1A", "SON") | 
                          (data$structure_name %in% c("TUBA1B", "LMNB1") & data$WorkflowId == "['Pipeline 4.4']") | 
                          (data$structure_name == "ATP2A2" & data$ImagingMode == "Mode C"),
                        "new_matrigel", "old_matrigel")

data$piezo <- ifelse(data$WorkflowId == "['Pipeline 4.4']", 
                     "triggered", "interactive")

# Select data - exclude (NPM1, RAB5, SLC25A17) from analysis because n is too low
data_select = subset(data, 
                     # (ImagingMode == 'Mode A')
                     # & (WorkflowId == "['Pipeline 4.4']")
                     # (protocol == "old_matrigel")
                     # & (InstrumentId %in% c("ZSD-1", "ZSD-2"))
                     # (structure_name %in% c("ATP2A2", "DSP", "HIST1H2BJ", "MYH10", "NUP153", "SON", "SMC1A", "ST6GAL1", "TJP1", "TOMM20")))
                     !(structure_name %in% c("NPM1", "RAB5A", "ATP2A2", "SLC25A17")))

# add random column
random_var <- data.frame(rnorm(nrow(data), mean=0, sd=1))
data_select = merge(data_select, random_var, by.x=0, by.y=0)
data_select <- rename(data_select,
                      c(
                        "rnorm.nrow.data...mean...0..sd...1."="random_variable"
                      )
)

# Normalize continuous variables
data_scaled <- as.data.frame(scale(data_select[cont_var]))

data_scaled <- rename(data_scaled, 
                      c(
                        "meta_confluency"="z_norm_meta_confluency", 
                        "meta_colony_area"="z_norm_meta_colony_area",
                        "meta_fov_edgedist"="z_norm_meta_fov_edgedist",
                        "meta_passage_post_thaw"="z_norm_meta_passage_post_thaw",
                        "meta_passage_total"="z_norm_meta_passage_total"
                        # "fov_median_mem_position_depth_lcc"="z_norm_fov_median_mem_position_depth_lcc"
                      )
)

# Merge normalized data
data_norm = merge(data_select, data_scaled, by.x=0, by.y=0)

# Drop NA rows in dataframe 
data_clean = data_norm[rowSums(is.na(data_norm[input_var])) == 0, ]

# Hold out 10 images for validation
val <- data.frame()
for (struc in unique(data_clean$structure_name)){
  data_struc <- subset(data_clean, (structure_name == struc))
  val_ind <- sample.int(n=nrow(data_struc), size=10)
  val_struc <- data_struc[val_ind, ]
  val <- rbind(val, val_struc)
}

output_var="fov_mem_position_depth"
input_var = c(
  "z_norm_meta_confluency",
  "z_norm_meta_colony_area",
  "z_norm_meta_fov_edgedist",
  "z_norm_meta_passage_post_thaw",
  "z_norm_meta_passage_total",
  "structure_name",
  "InstrumentId",
  "protocol",
  "piezo",
  "WorkflowId",
  "random_variable")

proj_folder = "//allen/aics/microscopy/Data/fov_in_colony"
dir.create(file.path(proj_folder, model_name), showWarnings=FALSE)

# Iterate through variables to make models. For each set of variables, repat n times to get CIs

iter_df = read.csv("\\\\allen\\aics\\microscopy\\Data\\fov_in_colony\\model_iterations.csv")
colnames(iter_df)

for (i in 1:nrow(iter_df)){
  model_name = iter_df[i, "model_name"]
  model_var_str = substr(iter_df[i, "model_variables"], 3, nchar(iter_df[i, "model_variables"])-2)
  input_var = unlist(strsplit(model_var_str, "', '"))
  
  data_performance = data.frame()
  data_feat = data.frame(matrix(ncol=length(input_var), nrow=num_iteration))
  colnames(data_feat) <- input_var
  data_pred = data.frame()
  
  for (i in 1:num_iteration){
    train <- data.frame()
    test <- data.frame()
    sampled_test <- data.frame()
    for (struc in unique(data_clean$structure_name)){
      data_struc <- subset(data_clean, (structure_name == struc))
      train_ind <- sample.int(n=nrow(data_struc), size=90)
      train_struc <- data_struc[train_ind, ]
      test_struc <- data_struc[-train_ind, ]
      sampled_test_struc <- test_struc[sample(nrow(test_struc), 20), ]
      
      train <- rbind(train, train_struc)
      test <- rbind(test, test_struc)
      sampled_test <- rbind(sampled_test, sampled_test_struc)
    }
    
    train$structure_name <- as.factor(train$structure_name)
    test$structure_name <- as.factor(test$structure_name)
    val$structure_name <- as.factor(val$structure_name)
    sampled_test$structure_name <- as.factor(sampled_test$structure_name)
    data_clean$structure_name <- as.factor(data_clean$structure_name)
    
    rf = randomForest(x=train[input_var],
                      y=train[[output_var]],
                      ntree=500,
                      importance=TRUE,
    )
    
    y_pred_sampled = predict(rf, newdata=sampled_test[input_var])
    y_pred = predict(rf, newdata=data_clean[input_var])
    y_pred_val = predict(rf, newdata=val[input_var])
    rf_targ_pred = data.frame(y_pred_val, val[[output_var]], i, rownames(val))
    data_pred <- rbind(data_pred, rf_targ_pred)
    
    rsq_func <- function (x, y) cor(x, y) ^ 2
    rsq_sampled = rsq_func(y_pred_sampled, sampled_test[[output_var]])
    rsq = rsq_func(y_pred, data_clean[[output_var]])

    mse = rf$mse[500]
    perf_df <- data.frame(mse, rsq, rsq_sampled)
    data_performance <- rbind(data_performance, perf_df)
    
    rf_feat_imp = as.data.frame(importance(rf))

    for (feat in row.names(rf_feat_imp)) {
      data_feat[i, feat] = rf_feat_imp[feat, "%IncMSE"]
    }
  }
  dir.create(file.path(proj_folder, model_name), showWarnings=FALSE)
  write.csv(data_performance, file.path(proj_folder, model_name, "bootstrap_perf.csv"))
  write.csv(data_feat, file.path(proj_folder, model_name, "bootstrap_feat.csv"))
  write.csv(data_pred, file.path(proj_folder, model_name, "bootstrap_targ_pred.csv"))
}
