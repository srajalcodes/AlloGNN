suppressMessages(library(frustratometeR))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) quit(status = 1)
pdb_file <- args[1]
out_dir <- args[2]
pdb_id <- args[3]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
tryCatch({
  conf_frust <- calculate_frustration(PdbFile = pdb_file, Mode = "configurational", Graphics = FALSE, Visualization = FALSE)
  write.csv(conf_frust$FrustrationData, file.path(out_dir, paste0(pdb_id, "_conf.csv")), row.names = FALSE)
}, error = function(e) { message(paste("Configurational error:", e$message)) })
tryCatch({
  mut_frust <- calculate_frustration(PdbFile = pdb_file, Mode = "mutational", Graphics = FALSE, Visualization = FALSE)
  write.csv(mut_frust$FrustrationData, file.path(out_dir, paste0(pdb_id, "_mut.csv")), row.names = FALSE)
}, error = function(e) { message(paste("Mutational error:", e$message)) })
