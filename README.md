# GeoSafe — deployment package

## Project structure

```text
GeoSafe_Deploy/
├── app.py
├── requirements.txt
└── data/
    ├── geosafe_clustered_incidents.csv
    ├── geosafe_hotspots.csv
    └── geosafe_model_metrics.csv
```

## Deploy to Streamlit Community Cloud

1. Create/sign in to a GitHub account at https://github.com/.
2. Create a repository (for example, `geosafe-women-safety`).
3. Upload `app.py`, `requirements.txt`, and the entire `data/` folder, preserving the names and structure above.
4. Open https://share.streamlit.io/ and connect GitHub if prompted.
5. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
6. Deploy. When the app starts successfully, copy the generated `https://...streamlit.app` URL and send it to your teacher.

Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Test locally first (optional)

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Important limitations

- This is an academic prototype, not a certified emergency-response or personal-safety service.
- The incident data is historical, not a live feed; route scores are experimental estimates, not probabilities of harm.
- The SOS interface does not automatically contact police or send messages. Test its behavior on the actual phone.
- Route lookup/geocoding require internet access and depend on public services.
- Before publishing, ensure the CSV files contain no personal, identifying, or otherwise confidential information.
