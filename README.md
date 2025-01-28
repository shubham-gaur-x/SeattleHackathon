# SeattleHackathon

Welcome to the **SeattleHackathon** project repository! This project was developed during the "Needle in a Hack Stack" hackathon hosted by Northeastern University in collaboration with the City of Seattle. The hackathon aimed to utilize Seattle's Building Energy Benchmarking Open Data to devise innovative solutions for advancing energy efficiency and reducing greenhouse gas emissions (GHG).

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Objective](#objective)
3. [Technologies Used](#technologies-used)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
6. [How to Run](#how-to-run)
7. [Future Improvements](#future-improvements)
8. [Acknowledgments](#acknowledgments)

---

## Project Overview
The "Needle in a Hack Stack" hackathon focused on leveraging Seattle's open data to identify actionable strategies for reducing building-related GHG emissions. The hackathon aligns with the City of Seattle's Building Emissions Performance Standard (BEPS) policy, aimed at reducing emissions from large buildings by 27% by 2050.

Participants were encouraged to:
- Analyze the Building Energy Benchmarking Open Data.
- Incorporate equity analysis using the Seattle Race and Social Equity Index.
- Develop solutions that address sustainability and equity challenges.

---

## Objective
The primary objective of this project is to harness public data to propose solutions for improving energy efficiency in Seattle's buildings. By integrating data analytics and machine learning, the project aims to:
- Identify trends in energy usage.
- Suggest actionable steps for reducing emissions.
- Ensure equitable outcomes for all communities.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**: 
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib/Seaborn (for visualization)
  - Flask (for web application, if applicable)
- **Tools**:
  - Jupyter Notebook
  - Joblib (for model serialization)

---

## Dataset
- **Source**: [Seattle Building Energy Benchmarking Open Data](https://data.seattle.gov/).
- **Description**: This dataset includes energy performance data for non-residential and multifamily properties (20,000 square feet or larger) in Seattle. Key metrics include energy usage, emissions, and building characteristics.
- **Preprocessing**: Data cleaning involved handling missing values, feature engineering, and merging additional datasets like the Seattle Race and Social Equity Index.

---

## Methodology
1. **Exploratory Data Analysis (EDA)**:
   - Visualizations to understand data distributions and trends.
   - Insights into correlations between features such as energy usage and emissions.

2. **Feature Engineering**:
   - Integration of equity data to identify disparities in energy performance.

3. **Model Training**:
   - Models used: Logistic Regression, K-Nearest Neighbors (KNN), etc.
   - Evaluation based on accuracy, precision, recall, and F1-score.

4. **Deployment (if applicable)**:
   - Flask app to showcase data insights and solutions interactively.

---

## How to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shubham-gaur-x/SeattleHackathon.git
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   Open `HackPOC.ipynb` or `HackPOCTest.ipynb` in Jupyter Notebook.

4. **Run the Flask App (if applicable)**:
   ```bash
   python app.py
   ```
   Navigate to `http://127.0.0.1:5000` in your browser to view the app.

---

## Future Improvements
- Incorporate additional datasets to enhance model predictions.
- Expand the analysis to include real-time energy data.
- Develop a user-friendly dashboard for interactive visualizations.
- Refine equity analysis to ensure more targeted recommendations.

---

## Acknowledgments
Special thanks to Northeastern University and the City of Seattle for organizing the hackathon and providing mentorship. The project is inspired by Seattle's commitment to sustainability and equity through initiatives like the Building Emissions Performance Standard (BEPS) policy and the One Seattle Data Strategy.

Kudos to all hackathon participants for their innovative solutions and collaborative efforts!
