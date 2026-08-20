#!/usr/bin/env Rscript

# Avoid harmless timezone warnings that can appear on some Windows/Conda setups.
Sys.setenv(TZ = "UTC")

args <- commandArgs(trailingOnly = TRUE)

start_year <- if (length(args) >= 1) as.integer(args[[1]]) else 2009
end_year   <- if (length(args) >= 2) as.integer(args[[2]]) else 2022
root       <- if (length(args) >= 3) args[[3]] else "data/raw/college"
force      <- if (length(args) >= 4) tolower(args[[4]]) %in% c("1", "true", "yes") else FALSE

required <- c("hoopR", "arrow", "dplyr")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing) > 0) {
  stop(
    paste0(
      "Missing R packages: ", paste(missing, collapse = ", "), "\n",
      "Install them before continuing."
    ),
    call. = FALSE
  )
}

if (utils::packageVersion("hoopR") < "3.1.0") {
  stop(
    paste0(
      "hoopR >= 3.1.0 is required for load_mbb_player_core().\n",
      "Installed version: ", utils::packageVersion("hoopR"), "\n",
      "Update with:\n",
      "install.packages('hoopR', repos=c(",
      "'https://sportsdataverse.r-universe.dev', ",
      "'https://cloud.r-project.org'))"
    ),
    call. = FALSE
  )
}

subdirs <- c(
  "player_box",
  "player_core",
  "player_season_from_box",
  "team_stats"
)

for (subdir in subdirs) {
  dir.create(file.path(root, subdir), recursive = TRUE, showWarnings = FALSE)
}


parquet_has_rows <- function(path) {
  if (!file.exists(path)) {
    return(FALSE)
  }

  ok <- tryCatch(
    {
      x <- arrow::read_parquet(path, as_data_frame = TRUE)
      nrow(x) > 0
    },
    error = function(e) FALSE
  )

  isTRUE(ok)
}


save_parquet <- function(tbl, path) {
  if (is.null(tbl) || nrow(tbl) == 0) {
    stop("Refusing to save a zero-row dataset: ", path, call. = FALSE)
  }

  arrow::write_parquet(tbl, path, compression = "zstd")
  message(sprintf("SAVE %s rows -> %s", format(nrow(tbl), big.mark = ","), path))
}


collect_one <- function(year, name, loader) {
  path <- file.path(root, name, sprintf("%s_%d.parquet", name, year))
  failed_path <- paste0(path, ".failed.txt")

  # Important: the old collector may have saved zero-row Parquets after a 404.
  # Treat those as invalid instead of skipping them.
  if (file.exists(path) && !force) {
    if (parquet_has_rows(path)) {
      message(sprintf("SKIP %s", path))
      return(arrow::read_parquet(path, as_data_frame = TRUE))
    } else {
      message(sprintf("REMOVE invalid/empty parquet: %s", path))
      file.remove(path)
    }
  }

  max_attempts <- 4

  for (attempt in seq_len(max_attempts)) {
    message(sprintf("GET  %s season=%d [attempt %d/%d]",
                    name, year, attempt, max_attempts))

    warning_messages <- character()

    result <- tryCatch(
      withCallingHandlers(
        loader(year),
        warning = function(w) {
          warning_messages <<- c(warning_messages, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      ),
      error = function(e) e
    )

    if (!inherits(result, "error") && !is.null(result) && nrow(result) > 0) {
      save_parquet(result, path)

      if (file.exists(failed_path)) {
        file.remove(failed_path)
      }

      return(result)
    }

    if (inherits(result, "error")) {
      message("ERROR: ", conditionMessage(result))
    } else {
      message("ERROR: loader returned zero rows.")
    }

    if (length(warning_messages) > 0) {
      message("Warnings:")
      for (msg in unique(warning_messages)) {
        message("  - ", msg)
      }
    }

    if (attempt < max_attempts) {
      Sys.sleep(2 ^ attempt)
    }
  }

  writeLines(
    c(
      sprintf("Collection failed for %s season %d.", name, year),
      "The loader returned an error or zero rows."
    ),
    failed_path
  )

  NULL
}


derive_player_season_from_box <- function(player_box, year) {
  if (is.null(player_box) || nrow(player_box) == 0) {
    return(NULL)
  }

  # Keep one row per athlete-team-season.
  # A transfer player can therefore have multiple rows in the same season;
  # that is useful information and can be collapsed later when building features.
  out <- player_box |>
    dplyr::mutate(
      appeared = !dplyr::coalesce(.data$did_not_play, FALSE),
      started = dplyr::coalesce(.data$starter, FALSE)
    ) |>
    dplyr::group_by(
      .data$season,
      .data$athlete_id,
      .data$athlete_display_name,
      .data$team_id,
      .data$team_display_name,
      .data$athlete_position_name,
      .data$athlete_position_abbreviation
    ) |>
    dplyr::summarise(
      games_on_roster = dplyr::n(),
      games_played = sum(.data$appeared, na.rm = TRUE),
      games_started = sum(.data$started & .data$appeared, na.rm = TRUE),

      minutes = sum(.data$minutes, na.rm = TRUE),

      field_goals_made = sum(.data$field_goals_made, na.rm = TRUE),
      field_goals_attempted = sum(.data$field_goals_attempted, na.rm = TRUE),

      three_point_field_goals_made =
        sum(.data$three_point_field_goals_made, na.rm = TRUE),
      three_point_field_goals_attempted =
        sum(.data$three_point_field_goals_attempted, na.rm = TRUE),

      free_throws_made = sum(.data$free_throws_made, na.rm = TRUE),
      free_throws_attempted = sum(.data$free_throws_attempted, na.rm = TRUE),

      offensive_rebounds = sum(.data$offensive_rebounds, na.rm = TRUE),
      defensive_rebounds = sum(.data$defensive_rebounds, na.rm = TRUE),
      rebounds = sum(.data$rebounds, na.rm = TRUE),
      assists = sum(.data$assists, na.rm = TRUE),
      steals = sum(.data$steals, na.rm = TRUE),
      blocks = sum(.data$blocks, na.rm = TRUE),
      turnovers = sum(.data$turnovers, na.rm = TRUE),
      fouls = sum(.data$fouls, na.rm = TRUE),
      points = sum(.data$points, na.rm = TRUE),

      .groups = "drop"
    ) |>
    dplyr::mutate(
      fg_pct = dplyr::if_else(
        .data$field_goals_attempted > 0,
        .data$field_goals_made / .data$field_goals_attempted,
        NA_real_
      ),
      three_pct = dplyr::if_else(
        .data$three_point_field_goals_attempted > 0,
        .data$three_point_field_goals_made /
          .data$three_point_field_goals_attempted,
        NA_real_
      ),
      ft_pct = dplyr::if_else(
        .data$free_throws_attempted > 0,
        .data$free_throws_made / .data$free_throws_attempted,
        NA_real_
      ),
      efg_pct = dplyr::if_else(
        .data$field_goals_attempted > 0,
        (
          .data$field_goals_made +
            0.5 * .data$three_point_field_goals_made
        ) / .data$field_goals_attempted,
        NA_real_
      ),
      ts_pct = dplyr::if_else(
        2 * (
          .data$field_goals_attempted +
            0.44 * .data$free_throws_attempted
        ) > 0,
        .data$points /
          (
            2 * (
              .data$field_goals_attempted +
                0.44 * .data$free_throws_attempted
            )
          ),
        NA_real_
      ),

      points_per_40 = dplyr::if_else(
        .data$minutes > 0,
        40 * .data$points / .data$minutes,
        NA_real_
      ),
      rebounds_per_40 = dplyr::if_else(
        .data$minutes > 0,
        40 * .data$rebounds / .data$minutes,
        NA_real_
      ),
      assists_per_40 = dplyr::if_else(
        .data$minutes > 0,
        40 * .data$assists / .data$minutes,
        NA_real_
      ),
      steals_per_40 = dplyr::if_else(
        .data$minutes > 0,
        40 * .data$steals / .data$minutes,
        NA_real_
      ),
      blocks_per_40 = dplyr::if_else(
        .data$minutes > 0,
        40 * .data$blocks / .data$minutes,
        NA_real_
      ),
      turnovers_per_40 = dplyr::if_else(
        .data$minutes > 0,
        40 * .data$turnovers / .data$minutes,
        NA_real_
      ),

      assist_turnover_ratio = dplyr::if_else(
        .data$turnovers > 0,
        .data$assists / .data$turnovers,
        NA_real_
      ),

      three_attempt_rate = dplyr::if_else(
        .data$field_goals_attempted > 0,
        .data$three_point_field_goals_attempted /
          .data$field_goals_attempted,
        NA_real_
      ),

      free_throw_rate = dplyr::if_else(
        .data$field_goals_attempted > 0,
        .data$free_throws_attempted /
          .data$field_goals_attempted,
        NA_real_
      )
    )

  path <- file.path(
    root,
    "player_season_from_box",
    sprintf("player_season_from_box_%d.parquet", year)
  )

  save_parquet(out, path)
  out
}


for (year in seq.int(start_year, end_year)) {
  message("")
  message("============================================================")
  message(sprintf("COLLEGE SEASON %d", year))
  message("============================================================")

  # 1) Main source of player performance.
  player_box <- collect_one(
    year,
    "player_box",
    function(y) hoopR::load_mbb_player_box(seasons = y)
  )

  # 2) Derive season stats ourselves.
  if (!is.null(player_box) && nrow(player_box) > 0) {
    derive_path <- file.path(
      root,
      "player_season_from_box",
      sprintf("player_season_from_box_%d.parquet", year)
    )

    if (force || !parquet_has_rows(derive_path)) {
      derive_player_season_from_box(player_box, year)
    } else {
      message(sprintf("SKIP %s", derive_path))
    }
  }

  # 3) Identity + bio. This replaces the old load_mbb_rosters() call.
  collect_one(
    year,
    "player_core",
    function(y) hoopR::load_mbb_player_core(seasons = y)
  )

  # 4) Team context. This worked in the original test and is still useful.
  collect_one(
    year,
    "team_stats",
    function(y) hoopR::load_mbb_team_stats(seasons = y)
  )
}

message("")
message("Done.")
