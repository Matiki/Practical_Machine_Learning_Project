# Load necessary R packages into current R session
library(readr)

# Check if data files exist in working directory, if not: download the data
if(!file.exists("pml-training.csv")){
        URL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
        download.file(URL,
                      destfile = "pml-training.csv")
        rm(URL)
}

if(!file.exists("pml-testing.csv")){
        URL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
        download.file(URL,
                      destfile = "pml-testing.csv")
        rm(URL)
}

# Read the data into R
training <- read_csv("pml-training.csv")
testing <- read_csv("pml-testing.csv")
