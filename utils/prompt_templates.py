def zero_shot_prompt(text):
    """
    Generate zero-shot prompt for sentiment analysis
    """
    return f"""Analyze the sentiment of the following movie review. 
    Respond with only one word: either 'positive' or 'negative'.
    
    Review: {text}
    
    Sentiment:"""

def few_shot_prompt(text):
    """
    Generate few-shot prompt for sentiment analysis
    """
    return f"""Analyze the sentiment of movie reviews. Here are some examples:

    Review: "This movie was amazing! The acting was superb and the story kept me engaged throughout."
    Sentiment: positive

    Review: "Terrible waste of time. Poor acting and confusing plot."
    Sentiment: negative

    Review: "A masterpiece of modern cinema. Every scene was perfectly crafted."
    Sentiment: positive

    Now analyze this review:
    Review: {text}
    
    Sentiment:"""
