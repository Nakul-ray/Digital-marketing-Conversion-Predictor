# Digital Marketing Conversion Prediction - Streamlit App

## Files needed in the same folder

- `app.py`
- `xgb_model.pkl`
- `scaler.pkl`
- `digital_marketing_campaign_dataset1.csv`
- `requirements.txt`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Create a GitHub repository.
2. Upload all required files to the repository.
3. Go to Streamlit Cloud.
4. Choose the GitHub repository.
5. Set the main file path as:

```text
app.py
```

6. Click Deploy.

## Notes

The app uses the saved XGBoost model and scaler. The model expects these 18 columns:

```text
Age, Income, AdSpend, ClickThroughRate, ConversionRate, WebsiteVisits,
PagesPerVisit, TimeOnSite, SocialShares, EmailOpens, EmailClicks,
PreviousPurchases, LoyaltyPoints, Gender_Male, CampaignChannel_PPC,
CampaignChannel_Referral, CampaignChannel_SEO, CampaignChannel_Social Media
```
