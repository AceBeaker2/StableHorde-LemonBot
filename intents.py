#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 23:36:30 2023

@author: oriontomasi
"""

#import fasttext
#from scipy import spatial

from itertools import product
from word_forms.word_forms import get_word_forms

def unleet(word):
    LEET_TO_STANDARD = {'4': 'a', '8': 'b', '3': 'e', '6': 'g', '1': 'i',
                       '0': 'o', '5': 's', '7': 't', '2': 'z'}
    translation_table = str.maketrans(LEET_TO_STANDARD)
    return word.translate(translation_table)

def redefine_lists():
    with open('filter/sex.txt', 'r') as file:
        sex_words = file.read()
        sex_words = sex_words.split('\n')
        sex_words = list(set(sex_words))
    
    with open('filter/young.txt', 'r') as file:
        young_words = file.read()
        young_words = young_words.split('\n')
        young_words = list(set(young_words))
        
    return sex_words, young_words

def check_intent(prompt, words):
    triggered = False
    
    word_alts = []
    triggers = []
    
    for word in prompt.split(' '):
        word_alts.append(word)
        
        unleetword = unleet(word)
        
        if not word == unleetword:
            word_alts.append(unleetword)
        
        forms = get_word_forms(word)
        for form in forms:
            word_alts.extend(list(forms[form]))
            
    word_alts.append(prompt)
    
    unleetprompt = unleet(prompt)
    word_alts.append(unleetprompt)
    
    for naughty_word in words:
        if naughty_word in prompt:
            triggered = True
            triggers.append(naughty_word)
            
        for word in word_alts:
            if naughty_word == word:
                triggered = True
                triggers.append(naughty_word)
    return (triggered, triggers)

def check_cp(prompt):
    sex_words, young_words = redefine_lists()
    c = check_intent(prompt, young_words)
    p = check_intent(prompt, sex_words)
    return c[0] and p[0], c[1], p[1]

if __name__ == '__main__':
    while 1:
        prompt = input('\nPrompt: ')
        print(check_cp(prompt))

