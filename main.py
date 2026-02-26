from  fastapi import FastAPI
from pydantic import  BaseModel
import  uvicorn
import  joblib


model = joblib.load('log_model.pkl')
scaler = joblib.load('scaler1.pkl')



bank_app = FastAPI()


class BankShema(BaseModel):
    person_age: float
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: int
    person_home_ownership: str
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    loan_percent_income:float
    cb_person_cred_hist_length: float
    credit_score: int
    previous_loan_defaults_on_file: str

@bank_app.post('/predict/', response_model=dict)
async def predict_bank(bank: BankShema):
    bank_dict = bank.dict()

    person_gender = bank_dict.pop('person_gender')
    person_gender_1_0 = [1 if person_gender == 'male' else 0]


    person_education = bank_dict.pop('person_education')
    person_education_1_0 =[
        1 if person_education == 'Bachelor' else 0,
        1 if person_education == 'Doctorate' else 0,
        1 if person_education == 'High School' else 0,
        1 if person_education == 'Master' else 0,]

    person_home_ownership = bank_dict.pop('person_home_ownership')
    person_home_ownership_1_0 = [
        1 if person_home_ownership == 'OTHER' else 0,
        1 if person_home_ownership == 'OWN' else 0,
        1 if person_home_ownership == 'RENT' else 0,
    ]

    loan_intent = bank_dict.pop('loan_intent')
    loan_intent_1_0 = [
        1 if loan_intent == 'EDUCATION' else 0,
        1 if loan_intent == 'HOMEIMPROVEMENT' else 0,
        1 if loan_intent == 'MEDICAL' else 0,
        1 if loan_intent == 'PERSONAL' else 0,
        1 if loan_intent == 'VENTURE' else 0,
    ]

    previous = bank_dict.pop('previous_loan_defaults_on_file')
    previous_1_0 = [
        1 if previous == 'Yes' else 0
    ]

    bank_data = (list(bank_dict.values()) +
     person_gender_1_0 + person_education_1_0 +
     person_home_ownership_1_0 + loan_intent_1_0 + previous_1_0)

    scaled_data = scaler.transform([bank_data])
    pred = model.predict(scaled_data)[0]
    new_pred = 'Approved' if pred == 1 else 'Rejected'
    return {'predict': new_pred}

if __name__ == '__main__':
    uvicorn.run(bank_app, host='127.0.0.1', port=8000)
