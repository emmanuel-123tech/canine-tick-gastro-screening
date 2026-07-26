# CanineCare AI

Streamlit research prototype for preliminary screening between the clinic-recorded classes of tick fever and gastroenteritis in dogs.

## Local run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Community Cloud

1. Upload this folder to a GitHub repository.
2. Confirm that `model_outputs/tick_fever_gastroenteritis_model_bundle.joblib` is included.
3. Create a new Streamlit Community Cloud app.
4. Select `streamlit_app.py` as the entry point.

The application is an academic prototype. It does not provide a veterinary diagnosis or treatment recommendation.
