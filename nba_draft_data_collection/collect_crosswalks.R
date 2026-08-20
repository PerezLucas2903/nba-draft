#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

college_start <- if (length(args) >= 1) as.integer(args[[1]]) else 2009
college_end   <- if (length(args) >= 2) as.integer(args[[2]]) else 2022
nba_end       <- if (length(args) >= 3) as.integer(args[[3]]) else college_end + 2
root          <- if (length(args) >= 4) args[[4]] else "data/raw/crosswalks"
force         <- if (length(args) >= 5) tolower(args[[5]]) %in% c("1", "true", "yes") else FALSE

required <- c("hoopR", "arrow")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing) > 0) {
  stop(
    paste0(
      "Missing R packages: ", paste(missing, collapse = ", "), "\n",
      "Run: Rscript install_r_packages.R"
    ),
    call. = FALSE
  )
}

dir.create(root, recursive = TRUE, showWarnings = FALSE)

save_if_needed <- function(path, loader) {
  if (file.exists(path) && !force) {
    message("SKIP ", path)
    return(invisible(NULL))
  }

  max_attempts <- 4

  for (attempt in seq_len(max_attempts)) {
    result <- tryCatch(loader(), error = function(e) e)

    if (!inherits(result, "error")) {
      arrow::write_parquet(result, path, compression = "zstd")
      message(sprintf("SAVE %s rows -> %s", nrow(result), path))
      return(invisible(result))
    }

    message(
      sprintf(
        "Attempt %d/%d failed for %s: %s",
        attempt, max_attempts, path, conditionMessage(result)
      )
    )

    if (attempt < max_attempts) {
      Sys.sleep(2 ^ attempt)
    }
  }

  writeLines(
    paste0("Collection failed after ", max_attempts, " attempts."),
    paste0(path, ".failed.txt")
  )
}

# College crosswalk: ESPN athlete IDs and other provider IDs.
mbb_path <- file.path(root, "mbb_player_crosswalk.parquet")
save_if_needed(
  mbb_path,
  function() {
    hoopR::load_mbb_player_crosswalk(
      seasons = seq.int(college_start, college_end)
    )
  }
)

# NBA crosswalk: crucial bridge from ESPN athlete ID to NBA.com PLAYER_ID.
nba_path <- file.path(root, "nba_player_crosswalk.parquet")
save_if_needed(
  nba_path,
  function() {
    hoopR::load_nba_player_crosswalk(
      seasons = seq.int(college_start, nba_end)
    )
  }
)

message("Done.")
