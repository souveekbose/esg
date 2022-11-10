import pandas as pd


def compute_social_harm():
    social_news = pd.read_csv('./resources/social.csv')
    print(social_news)

