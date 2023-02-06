"""
the point of this file is to help the main model handle with entities in a sentence if they dont appear in its vocab such as:
duration, phone number, names, locations, emails, dates, time, url and currency.
the way it will be implemented is using an existing NER model combined with regular expressions.
the reason it will be implemented using an existing NER model is due to an early experience in which a BERT model was fine tuned 
on the small corpus of data available online for the types of entities mentioned earlier and yield not great results for very likebale inputs from a user
therefore, it will be better to use an already existing NER model which is trained on very large corpus of data, much larger than the BERT model was fine tuned on
than trying to create a new NER model from scratch that will most likley achive even worst results than the fine-tuned BERT model.
the input from the user might look like this:
please send an email to example@gmail.com tommorow morning
it will be turned into something like:
please send an email to EMAIL_0 DATE_0
to help the main model handle "example@gmail.com" and "tommorow morning" and they will be replaced at the end if the main model will find them necessery for the output.
"""

import spacy
import en_core_web_trf
import re


def replace_string_entities(sentence : str, final_dictionary : dict, model_vocab : list):
    """
    this function takes the finished vocab that maps between the entities in the original sentence and their corresponding main model tag
    and replace them in the original sentence.
    :param sentence: the sentence to replace the tags in
    :type sentence: str
    :param final_dictionary: the dictionary that contains the tags and their corresponding word
    :type: dict
    :return: the sentence with the replaced entities
    :rtype: str
    """
    for (key, value) in final_dictionary.items():
    #  the if statment is there for cases that a value appears twice in the dictionary (because it was guessed by both the NER and RE, thats why the order of the processing of the sentence is important too) 
    #  and also to make sure that its only replacing words that do not appear in the main model vocab because if they do appear the model will know how to handle them
    #  and it probably means that they are part of the main vocab for a reason, for example twitter might be a web service used in the main model and in case of the query
    #  "post picture on twitter" replacing twitter with NAME_0 might lead to bad results of the main model that will not be able to guess what web service to use, should it use twitter? or maybe instagram?
        if (value in sentence) and (value not in model_vocab):
            sentence = sentence.replace(value, key)
    return sentence


def extract_url_or_domain(sentence : str, final_dictionary : dict):
    """
    this function uses regular expressions to extract a domain or url and adds them to the main dictionary.
    :param sentence: the sentence to replace the tags in
    :type sentence: str
    :param final_dictionary: the dictionary that contains the tags and their corresponding word
    :type: dict
    """
    domains = re.findall(r'((?:https?:)?//)?(www\.)?([A-Za-z0-9-]+\.[A-Za-z0-9-]+)(?:/[A-Za-z0-9./]+)?$', sentence)
    print(domains)
    for url in range(len(domains)):
        protocol = domains[url][0] if domains[url][0] else ''
        subdomain = domains[url][1] if domains[url][1] else ''
        domain = domains[url][2]
        final_dictionary[f"URL_{url}"] = protocol + subdomain + domain


def extract_phone_numbers(sentence : str, final_dictionary : dict):
    """
    this function uses regular expressions to extract a phone number and adds them to the main dictionary.
    :param sentence: the sentence to replace the tags in
    :type sentence: str
    :param final_dictionary: the dictionary that contains the tags and their corresponding word
    :type: dict
    """
    phone_numbers = re.findall(r'\s((?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})\s', sentence)
    for phone_index in range(len(phone_numbers)):
        final_dictionary[f"PHONE_{phone_index}"] = phone_numbers[phone_index]


def extract_emails(sentence : str, final_dictionary : dict):
    """
    this function uses regular expressions to extract an email address and adds them to the main dictionary.
    :param sentence: the sentence to replace the tags in
    :type sentence: str
    :param final_dictionary: the dictionary that contains the tags and their corresponding word
    :type: dict
    """
    emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', sentence)
    for email_index in range(len(emails)):
        final_dictionary[f"EMAIL_{email_index}"] = emails[email_index]


def ner_usage(sentence : str, final_dictionary : dict):
    """
    this function uses a ner model to extract entities like names or locations and adds them to the final dictionary that maps
    between tags to their corresponding entity. 
    :param sentence: the sentence to replace the tags in
    :type sentence: str
    :param final_dictionary: the dictionary that contains the tags and their corresponding word
    :type: dict
    """
    model = en_core_web_trf.load()
    processed_sentence = model(sentence)
    #  that dictionary contains the scapy tags as keys and the model tag and the count of times that tag appeard in the sentence 
    spacy_tag_to_model_tag = {
    ("PERSON", "ORG", "NORP", "WORK_OF_ART", "LANGUAGE", "PRODUCT"):["NAME_", 0],
    ("GPE", "FAC", "LOC"):["LOCATION_", 0],
    ("TIME"):["TIME_", 0],
    ("DATE", "EVENT"):["DATE_", 0],
    ("CARDINAL", "QUANTITY", "PERCENT", "ORDINAL"):["NUMBER_", 0],
    ("MONEY"):["CURRENCY_", 0]}
    for entity in processed_sentence.ents:
        for key in spacy_tag_to_model_tag.keys():
            if entity.label_ in key:
                final_dictionary[spacy_tag_to_model_tag[key][0] + str(spacy_tag_to_model_tag[key][1])] = entity.text
                spacy_tag_to_model_tag[key][1] += 1


def prepare_sentance(sentence : str, model_vocab : list):
    """
    this function will receive a natural language sentence and use regular expressions and NER model to turn the entities in it into more generic 
    types in case that they dont appear in the main model vocab. 
    for example, given the sentence "who is bill gates" this function will return "who is NAME_0" and a dictionary that maps between NAME_0 to "bill gates" 
    :param sentence: the sentence to replace the tags in
    :type sentence: str
    :param model_vocab: the main model vocabulary
    :type: list
    :return: the sentence with the replaced entities as their tags and a dictionary that maps between the tags to their entity
    :rtype: str
    """
    final_dictionary = {}
    extract_url_or_domain(sentence, final_dictionary)
    extract_phone_numbers(sentence, final_dictionary)
    extract_emails(sentence, final_dictionary)
    ner_usage(sentence, final_dictionary)
    return replace_string_entities(sentence, final_dictionary, model_vocab), final_dictionary