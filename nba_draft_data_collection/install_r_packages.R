#!/usr/bin/env Rscript

repos <- c(
  sportsdataverse = "https://sportsdataverse.r-universe.dev",
  CRAN = "https://cloud.r-project.org"
)

packages <- c("hoopR", "arrow")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing ", pkg, "...")
    install.packages(pkg, repos = repos)
  } else {
    message(pkg, " already installed.")
  }
}
