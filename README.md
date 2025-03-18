# CHLA No Show Predictor
In this project I create a deployed machine learning model to help CHLA predict patients that aren't likely to show up to their appointment. CHLA is the #1 ranked children’s hospital in California and the Pacific region according to U.S. News. As of 2023 they admitted 14,600 patients annually and employed more than 6,000 people. Serving so many patients CHLA inevitably ran into the issue of patients not showing up for appointments, wasting vital hospital resources and people’s time. In response, CHLA began double booking appointments to ensure there was a patient in each available time slot. However, this had an adverse effect as when patients did show up for their appointments, they often had longer wait times.
To solve this problem CHLA questioned whether it was possible to predict patient no-shows to more effectively double book appointments. If this were possible CHLA could better allocate their resources and plan for no shows in advance. By seeing which patients had a higher likelihood of not showing up CHLA could also implement intervening measures to encourage those high risk patients to either reschedule or arrive on time thus improving no show rates overall.

# Cloud deployed Link
https://chlapredictor.streamlit.app/

## Steps for Running the Code:
1. Clone the repository:
```bash
   git clone git@github.com:troppster225/CHLA_predictor
```
2. Install the following dependencies if not already installed:
* python 3.11.5
* pandas
* sklearn
* seaborn
* matplotlib
* streamlit
3. Open the no_show_predictor.ipynb
4. Run each cell and read the comments and markdown to understand what's going on
5. Go to your terminal
6. Ensure you are in the project directory
7. Enter the following command:
```bash
   streamlit run app.py
```
8. Play around with the local app!