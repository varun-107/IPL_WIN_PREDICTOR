import pandas as pd
import streamlit as st
import pickle



teams = [
         'Mumbai Indians',
         'Chennai Super Kings',
         'Royal Challengers Bangalore',
         'Delhi Capitals',
         'Rajasthan Royals',
         'Kings XI Punjab',
         'Kolkata Knight Riders',
         'Sunrisers Hyderabad'
]

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Cuttack', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl', 'rb'))
st.title(' ipl_win_predictor ')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', step=1, value=0, format='%d')

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score', step=1, value=0, format='%d')

with col4:
    overs = st.number_input('Overs completed', step=1, value=0, format='%d')

with col5:
    wickets = st.number_input('Wickets out', step=1, value=0, format='%d')

if st.button('Predict Probability'):

    runs_left = target - score

    balls_left = 120 - (overs*6)

    wickets = 10 - wickets

    crr = score/overs

    rr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city], 'runs_left' : [runs_left], 'balls_left': [balls_left], 'wickets': [wickets], 'total_runs_x': [target], 'crr': [crr], 'rr': [rr]})

    # st.table(input_df)
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.text(result)
    st.header(batting_team + "-" + str(round(win*100)) + "%")
    st.header(bowling_team + "-" + str(round(loss*100)) + "%")





