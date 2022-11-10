from transformers import BertTokenizer, BertForSequenceClassification, pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg',num_labels=4)
# finbert.save_pretrained("./models/finbert")
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')
# tokenizer.save_pretrained("./models/tokenizer")

esg_score = pipeline("text-classification", model=finbert, tokenizer=tokenizer)
esg_pillars = pipeline("text-classification", model="nbroad/ESG-BERT", top_k=None)
esg_pillars.save_pretrained("./models/tokenizer")

# Define the pillars
pillars = ['Social', 'Environment', 'Governance']

subPillars = {
    'Environment': ['Environment'],
    'Social': ['Customers', 'Human Rights & Community','Labor Rights & Supply Chain'],
    'Governance': ['Governance']
}

pillarIndicators = {
    'Environment': ['Physical_Impacts_Of_Climate_Change',
                    'Waste_And_Hazardous_Materials_Management',
                    'Water_And_Wastewater_Management',
                    'Air_Quality',
                    'Ecological_Impacts',
                    'Energy_Management',
                    'GHG_Emissions'],
    'Customers': ['Business_Ethics',
                  'Data_Security',
                  'Access_And_Affordability',
                  'Competitive_Behavior',
                  'Customer_Welfare',
                  'Product_Quality_And_Safety',
                  'Product_Design_And_Lifecycle_Management',
                  'Selling_Practices_And_Product_Labeling',
                  'Customer_Privacy'],
    'Human Rights & Community': ['Employee_Engagement_Inclusion_And_Diversity',
                                 'Human_Rights_And_Community_Relations',
                                 'Management_Of_Legal_And_Regulatory_Framework'],
    'Labor Rights & Supply Chain': ['Employee_Health_And_Safety',
                                    'Labor_Practices',
                                    'Supply_Chain_Management',
                                    'Systemic_Risk_Management'],
    'Governance': []
}


def compute_esg_score(raw_text_sanitized):
    # Determine the esg score of an article
    score = esg_score(raw_text_sanitized)
    print(score)
    if len(score) > 0 and score[0]['label'] != 'None':
        return score
    else:
        return 'None';

