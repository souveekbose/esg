from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

ner_bert = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
ner_bert.save_pretrained("./models/ner_bert")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

ner = pipeline("ner", model=model, tokenizer=ner_bert)


def is_corporate_esg(news_raw):
    print("checking if news has corporate esg")
    ner_output = ner(news_raw)
    orgs_in_data = [org['word'] for org in ner_output if org['entity'] in ['B-ORG','ORG', 'I-ORG']]
    if orgs_in_data:
        return True
    print(ner_output)


