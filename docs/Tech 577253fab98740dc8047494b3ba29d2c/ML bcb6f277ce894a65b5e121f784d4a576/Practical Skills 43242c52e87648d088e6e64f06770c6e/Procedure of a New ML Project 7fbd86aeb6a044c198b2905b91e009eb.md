# Procedure of a New ML Project

---

We will start by asking ourselves a bunch of questions. After answering all of them, our project design will be almost ready.

Also keep in mind, the following phrases are often iterating again and again. It’s different with traditional system development.

## Self-questioning

- What KIND of data needed
- Where is data from
- How much data we need
    - if not enough, what data augmentation skills are needed
- How will we LABEL the data (e.g. format, software…etc)
- The model will be served as REAL TIME or BATCH PROCESS
- What metrics to evaluate the model
    - e.g. For f-score, what f value should be used, it depends on the project focus.
- After launch, how often we will retrain the model

## Data collection:

## Explore data - EDA (Exploratory Data Analysis)

### Browse data :

check every column

analyze data structure or characteristic

distinguish between nominal & continuous field

of cause , mark the target/label field

Compare data balance between Train / Test / Eval

### What data is helpful and what is not :

describe the observation of EACH columns / data fields from the analysis above

decide what kind of preprocessing

## Feature Engineering

## Modeling