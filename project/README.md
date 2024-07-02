Introduction
In this project, we aimed to predict flight prices using a Random Forest regression model. We employed various preprocessing techniques to ensure the data was clean and suitable for modeling.


FastAPI 
use the curlcommand to test the endpoint:
curl --location 'http://127.0.0.1:5001/predict/' \
--header 'Content-Type: application/json' \
--data '{
    "Airline": "IndiGo",
    "Source": "Banglore",
    "Destination": "New Delhi",
    "Total_Stops": 0,
    "Date": 24,
    "Month": 3,
    "Year": 2019,
    "Dep_hours": 22,
    "Dep_min": 20,
    "Arrival_hours": 1,
    "Arrival_min": 10,
    "Duration_hours": 2,
    "Duration_min": 50
  }'