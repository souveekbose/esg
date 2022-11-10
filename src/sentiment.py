from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
classify = pipeline("text-classification", model=model, tokenizer=tokenizer)


def is_negative(raw_text_sanitized):
    result = classify(raw_text_sanitized)
    print(result)
    if result[0]['label'] in ['3 stars', '2 stars', '1 star', '0 star']:
        return result
    else:
        print('not negative')

