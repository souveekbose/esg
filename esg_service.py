import src.esg_identifier as esg_identifier
import src.sentiment as sentiment
import src.ner as ner
import src.nature_of_harm as nature


raw_text = '''
Microsoft lied to people
'''

# raw_text_sanitized = filtered_text

is_negative_news = sentiment.is_negative(raw_text)

if is_negative_news:
    esg_info = esg_identifier.compute_esg_score(raw_text)
    print(esg_info)
    if esg_info != 'None':
        is_corporate_esg = ner.is_corporate_esg(raw_text)
        if is_corporate_esg:
            print("corporate esg")
            nature.compute_social_harm()


def get_esg_response():
    esg = [
        {'Microsoft': '1'},
        {'Amazon': '1'}
    ]
    return esg
