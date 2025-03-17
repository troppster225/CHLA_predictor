import streamlit as st
import pickle
import datetime
import pandas as pd
import numpy as np

with open('noshow_predictor.sav', 'rb') as file:
    model = pickle.load(file)

def predict_noshow(model, features):
    prediction = model.predict([features])
    return "No Show" if prediction[0] == 1 else "Did not survive"
#Streamlit App
def main():
    st.title("CHLA No-Show Predictor")
    st.caption("Predict which appointments are likely to no-show with custom date range and clinic filtering.")

    # 1. User Inputs
    with st.form("my_form"):
        clinic_name = st.selectbox(
            "Select Clinic Name",
            [
                "BAKERSFIELD CARE CLINIC",
                "ENCINO CARE CENTER",
                "SANTA MONICA CLINIC",
                "SOUTH BAY CARE CENTER",
                "VALENCIA CARE CENTER",
                "ARCADIA CARE CENTER"
            ],
        )
        col1, col2 = st.columns(2)
        with col1:
            default_start = datetime.date(2024, 1, 1)
            start_date = st.date_input("Start Date", default_start)
        with col2:
            default_end = datetime.date(2024, 1, 15)
            end_date = st.date_input("End Date", default_end)

        if st.form_submit_button("Submit"):
        # input validation for ensuring that we have two dates
            if start_date > end_date:
                st.warning("Start date cannot be after end date.")
                st.stop()

            # 2. Loading our appointments df
            appointments_df = pd.read_csv("CHLA_clean_data_2024_Appointments.csv")
            
            # we first need to subset our df so we just get the data for our appointments range
            appointments_df["APPT_DATE"] = pd.to_datetime(
                appointments_df["APPT_DATE"], 
                format="%m/%d/%y %H:%M", 
                errors="coerce"
            )
            # creating a column with just the date and just time rather than date and time
            appointments_df["APPT_DATE_ONLY"] = appointments_df["APPT_DATE"].dt.date
            appointments_df["APPT_TIME"] = appointments_df["APPT_DATE"].dt.strftime("%H:%M")

            # filtering df for appointment dates given
            filtered_df = appointments_df[
                (appointments_df["CLINIC"].str.upper() == clinic_name.upper()) &
                (appointments_df["APPT_DATE_ONLY"] >= start_date) &
                (appointments_df["APPT_DATE_ONLY"] <= end_date)
            ].copy()

            # in case there are no appointments in that time
            if filtered_df.empty:
                st.warning("No appointments found for the given criteria.")
                return
            
            # 3. Now we do some data pre processing to get the dataset ready to be fed into our model

            #filtering for just the columns we want for our model

            feature_cols = ['LEAD_TIME', 'TOTAL_NUMBER_OF_CANCELLATIONS',
            'TOTAL_NUMBER_OF_RESCHEDULED',
            'TOTAL_NUMBER_OF_NOT_CHECKOUT_APPOINTMENT',
            'TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT', 'DAY_OF_WEEK', 'WEEK_OF_MONTH',
            'NUM_OF_MONTH', 'HOUR_OF_DAY', 'AGE', 'CLINIC',
            'IS_REPEAT', 'APPT_TYPE_STANDARDIZE']
            
            #columns we had for training that we want our filtered dataset to match
            
            trained_columns = ['LEAD_TIME', 'TOTAL_NUMBER_OF_CANCELLATIONS',
            'TOTAL_NUMBER_OF_RESCHEDULED',
            'TOTAL_NUMBER_OF_NOT_CHECKOUT_APPOINTMENT',
            'TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT', 'DAY_OF_WEEK', 'WEEK_OF_MONTH',
            'NUM_OF_MONTH', 'HOUR_OF_DAY', 'AGE', 'CLINIC_BAKERSFIELD CARE CLINIC',
            'CLINIC_ENCINO CARE CENTER', 'CLINIC_SANTA MONICA CLINIC',
            'CLINIC_SOUTH BAY CARE CENTER', 'CLINIC_VALENCIA CARE CENTER',
            'IS_REPEAT_Y', 'APPT_TYPE_STANDARDIZE_New',
            'APPT_TYPE_STANDARDIZE_Others']
            
            #filtering for features we want

            selected_df = filtered_df[feature_cols]

            #encoding
            categorical_cols = list(selected_df.select_dtypes(include=['object']).columns)
            encoded_df = pd.get_dummies(selected_df, columns=categorical_cols, drop_first=False)

            #reindexing to match training dataset
            encoded_df = encoded_df.reindex(columns=trained_columns, fill_value=0)

            #4. Getting our prediction and outputting for the user

            display_cols = ["MRN", "APPT_ID", "APPT_DATE_ONLY", "APPT_TIME"]
            display_df = filtered_df[display_cols].copy()
            display_df["MRN"] = display_df["MRN"].astype(str)
            display_df["APPT_ID"] = display_df["APPT_ID"].astype(str)

            probs = model.predict_proba(encoded_df)

            predicted_label = model.predict(encoded_df)

            predicted_no_show = np.where(predicted_label == 1, "Y", "N")

            probs_no_show = probs[:, 1]

            display_df["No Show"] = predicted_no_show
            display_df["Prob"] = probs_no_show

            st.session_state["prediction_results"] = display_df

            st.success("Prediction complete!")
    if "prediction_results" in st.session_state:
        st.subheader("Final Prediction Results")
        st.dataframe(st.session_state["prediction_results"])

if __name__ == "__main__":
    main()