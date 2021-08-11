from django.shortcuts import render
from django.forms import modelform_factory
from django.views.generic.base import TemplateView
from django.views.generic.list import ListView
from django.http import HttpResponse

from .models import Word

import json
import random

import spacy
from spacy.matcher import Matcher

import pandas as pd
from random import sample

import time
from collections import Counter

import ginza
import pykakasi
kks = pykakasi.wakati()

#articles = pd.read_csv(r'nlp/static/nlp/cleaned_articles_with_sentences.csv')
articles = pd.read_csv(r'nlp/static/nlp/cleaned_articles_with_sentences.csv')
#nlp = spacy.load('ja_core_news_lg')
#nlp = spacy.load('en_core_web_lg')
nlp = spacy.load('ja_ginza')
# nlp.add_pipe(nlp.create_pipe("merge_noun_chunks"))
# nlp.add_pipe(nlp.create_pipe("merge_entities"))
# nlp.add_pipe(nlp.create_pipe("merge_subtokens"))

class HomePageView(TemplateView):
    template_name = 'nlp/home.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

class TextPageView(TemplateView):
    template_name = 'nlp/text.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

def parse_matches(match_string):
    pattern = []
    str = ""
    for c in match_string:
        if (c == "n"):
            pattern.append({ "POS": "NOUN" })
            if (len(str) > 0):
                pattern.append({ "TEXT": str })
            str = ""
        elif (c == "v"):
            pattern.append({ "POS": "VERB" })
            if (len(str) > 0):
                pattern.append({ "TEXT": str })
            str = ""
        elif (c == "a"):
            pattern.append({ "POS": "ADJ" })
            if (len(str) > 0):
                pattern.append({ "TEXT": str })
            str = ""
        else:
            str += c
    if (len(str) > 0):
        pattern.append({ "TEXT": str })
    return pattern
            

class Match:
    def __init__(self, nlp, matcher, max_matches, const_words):
        self.matcher = matcher
        self.nlp = nlp
        self.max_matches = max_matches
        self.const_words = const_words
        self.matches = 0

    def contains_const_word(self, x):
        return len(self.const_words) == 0 or any([ _ in x for _ in self.const_words ])

    def get_doc(self, x):
        return self.nlp(x) 
    
    def n_matches(self):
        return self.matches >= self.max_matches

    def add_matches(self, n):
        self.matches += n
    
    def find_matches(self, x):
        if (self.n_matches() or not self.contains_const_word(x)): 
            return False
        doc = self.get_doc(x) 
        matches = self.matcher(doc)
        if (len(matches) > 0):
            self.add_matches(1)
            return True
        return False
    
def input_match(request):
    if request.method == 'POST':
        match = request.POST.get('match')
        matcher = Matcher(nlp.vocab)
        response_data = {}
        
        example_sentences = []
        nsents_per_match = 10 
        nmost_similar = 10

        pattern = parse_matches(match)
        const_words = [ _['TEXT'] for _ in pattern if 'TEXT' in _ ]
        matcher.add("match", None, pattern)
        matchObj = Match(nlp, matcher, nsents_per_match, const_words)
        
        for index in range(40):
            clean_column = f'clean_sent{index}'
            sent_column = f'sent{index}'
            found_rows = articles[clean_column].apply(lambda x: (not isinstance(x, float)) and matchObj.find_matches(x))
            nfound = found_rows.sum()
            articles.loc[found_rows].sample(min(nfound, nsents_per_match)).apply(
                lambda r: example_sentences.append(
                    { 
                        'word': '',
                        'sentence': r[sent_column],
                        'link': r['Links'],
                        'article': r['Text'],
                        'title': r['Title'],
                        'date': r['Date']
                    }
                ), axis=1    
            )

        response_data['example_sentences'] = example_sentences 


        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )
    else:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_Type="application/json"
        )

def input_text(request):
    if request.method == 'POST':
        text = request.POST.get('text')
        addWords = request.POST.getlist('addWords[]')
        subWords = request.POST.getlist('subWords[]')
        response_data = {}

        # nlp stuff
        similar_words = []
        if text:
            vectors = [ e.vector for e in nlp(text) ]
            vector = sum(vectors)
            most_similar = nlp.vocab.vectors.most_similar(
                vector.reshape(1, 300), n=100)
            similar_words = [nlp.vocab.strings[v]
                             for v in most_similar[0][0]]
        elif addWords or subWords:
            vector = None
            if (len(addWords) > 0 and len(subWords) > 0):
                addVector = sum([ nlp.vocab.get_vector(e) for e in addWords])
                subVector = sum([ nlp.vocab.get_vector(e) for e in subWords])
                vector = addVector - subVector
            elif (len(addWords) > 0):
                vector = sum([ nlp.vocab.get_vector(e) for e in addWords])
            else:
                vector = -sum([ nlp.vocab.get_vector(e) for e in subWords])

            most_similar = nlp.vocab.vectors.most_similar(
                vector.reshape(1, 300), n=100)
            similar_words = [nlp.vocab.strings[v]
                             for v in most_similar[0][0]]
        #nothing was found
        if (len(similar_words) > 0 and similar_words[0] == "ボリュームクッションラグ"):
            similar_words = []

        example_sentences = []
        nsents_per_word = 10 
        nmost_similar = 10
        sim_words = list(dict.fromkeys([text] + similar_words[:nmost_similar]))
        for word_ in sim_words:
            word = word_
            word_doc = nlp(word)
            if (len(word_doc) > 0 and word_doc[0].pos_ == 'VERB'):
                word = word_doc[0].lemma_ 
            for index in range(40):
                clean_column = f'clean_sent{index}'
                sent_column = f'sent{index}'
                found_rows = articles[clean_column].apply(lambda x: (not isinstance(x, float)) and word in x.split())
                nfound = found_rows.sum()
                articles.loc[found_rows].sample(min(nfound, nsents_per_word)).apply(
                    lambda r: example_sentences.append(
                        { 
                            'word': word,
                            'sentence': r[sent_column],
                            'clean_sentence': r[clean_column],
                            'link': r['Links'],
                            'article': r['Text'],
                            'title': r['Title'],
                            'date': r['Date']
                        }
                    ), axis=1    
                )

        response_data['example_sentences'] = example_sentences
        found_words = list(dict.fromkeys([ _['word' ] for _ in example_sentences ]))
        response_data['similar_words'] = [ _ for _ in found_words if not nlp(_)[0].is_stop] 


        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )
    else:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_Type="application/json"
        )


class VocabView(TemplateView):
    template_name = 'nlp/vocab.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context


def top_sentence(doc, limit):
    keyword = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.is_punct):
            continue
        if(token.pos_ in pos_tag):
            keyword.append(token.text)

    if (len(keyword) == 0): return [] 

    freq_word = Counter(keyword)
    max_freq = Counter(keyword).most_common(1)[0][1]
    for w in freq_word:
        freq_word[w] = (freq_word[w]/max_freq)
        
    sent_strength={}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent]+=freq_word[word.text]
                else:
                    sent_strength[sent]=freq_word[word.text]
    
    summary = []
    
    sorted_x = sorted(sent_strength.items(), key=lambda kv: kv[1], reverse=True)
    
    counter = 0
    for i in range(len(sorted_x)):
        summary.append(str(sorted_x[i][0]))

        counter += 1
        if(counter >= limit):
            break
            
    return ' '.join(summary)

def good_token(token):
    return (token.is_stop != True and token.is_punct != True and token.is_space != True)

nmost_common = 5 
def get_vocab(request):
    if request.method == 'POST':
        max_text_size = 1000000
        text = request.POST.get('text')
        matchRequest = request.POST.get('matchRequest')
        pattern = []
        matcher = Matcher(nlp.vocab)
        if (matchRequest == 'true'):
            try:
                pattern = json.loads(request.POST.get('match'))
                matcher.add("match", None, pattern)
            except:
                pass
        
        text = text[:max_text_size] 
        # originalText = text
        text = text.replace('\n', '')
        # text = text.replace('「', '')
        # text = text.replace('」', '')
        doc = nlp(text)

        matches = matcher(doc)
        match_results = []
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]
            match_results.append(span.text)

        
        ents = [ f"{token.text} ({token.label_.replace('_', ' ')})" for token in doc.ents ]

        words = [token.text for token in doc if good_token(token)] 

        nouns = [token.text for token in doc.noun_chunks]

        verbs = [token.lemma_ for token in doc if good_token(token) and token.pos_ == "VERB"]
        
        adjs = [token.lemma_ for token in doc if good_token(token) and token.pos_ == "ADJ"]

        word_freq = Counter(words)
        common_words = word_freq.most_common(nmost_common)

        noun_freq = Counter(nouns)
        common_nouns = noun_freq.most_common(nmost_common)

        verb_freq = Counter(verbs)
        common_verbs = verb_freq.most_common(nmost_common)
        
        adj_freq = Counter(adjs)
        common_adjs = adj_freq.most_common(nmost_common)
        
        ent_freq = Counter(ents)
        common_ents = ent_freq.most_common()

        #summary_len = round(0.1 * len(list(doc.sents)))
        summary_len = min(len(list(doc.sents)), 3) 
        summary = top_sentence(doc, summary_len)

        hira = {}
        fullText = ""
        for token in ginza.bunsetu_spans(doc):
            o = ""
            h = ""
            for item in kks.convert(token.text):
                o += item['orig']
                h += item['hira']
            fullText += o
            hira[o] = h 

        response_data = {
            'original': fullText, 
            'full': doc.to_json(),
            'hira': json.dumps(hira),
            'matchResult': match_results,
            'summary': summary,
            'words': common_words,
            'nouns': common_nouns,
            'verbs': common_verbs,
            'adjs': common_adjs,
            'ents': common_ents,
        }
                
                
        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )
    else:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_Type="application/json"
        )

def add_word(request):
    if request.method == 'POST':
        word_string = request.POST.get('word')
        word_string = word_string.replace('「', '')
        word_string = word_string.replace('」', '')
        word_string = nlp(word_string)[0].lemma_
        hira_string = request.POST.get('hira')
        ex_sent = request.POST.get('ex_sent')
        if not ex_sent: ex_sent = ''
        words = Word.objects.filter(word=word_string, hira=hira_string)
        exists = len(words) > 0
        word = Word(word='')
        if (not exists):
            word = Word(word=word_string, hira=hira_string, ex_sent=ex_sent, known=1.0) 
            word.save()
        else:
            word = words[0]

        response_data = { 
            'word': word.word,
        }
                
        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )
    else:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_Type="application/json"
        )

def make_tree(request):
    if request.method == 'POST':
        words = Word.objects.all()

        sizeMap = {}
        parentMap = {}
        root = 'hiragana'
        tree = [['hira', 'parent', 'size'], [root, '', 1]]
        for word in words:
            hira = "".join([ _['hira'] for _ in kks.convert(word.word)])
            if (len(hira) == 0):
                continue
            elif (len(hira) == 1):
                if (hira in sizeMap):
                    sizeMap[hira] += 1
                else:
                    sizeMap[hira] = 1
                parentMap[hira] = root 
            elif (len(hira) == 2):
                hira_first = hira[0]
                if (hira_first in sizeMap):
                    sizeMap[hira_first] += 1
                else:
                    sizeMap[hira_first] = 1
                hira_second = hira[:2]
                if (hira_second in sizeMap):
                    sizeMap[hira_second] += 1
                else:
                    sizeMap[hira_second] = 1
                parentMap[hira_first] = root
                parentMap[hira_second] = hira_first
            else:
                hira_first = hira[0]
                if (hira_first in sizeMap):
                    sizeMap[hira_first] += 1
                else:
                    sizeMap[hira_first] = 1
                hira_second = hira[:2]
                if (hira_second in sizeMap):
                    sizeMap[hira_second] += 1
                else:
                    sizeMap[hira_second] = 1
                if (hira in sizeMap):
                    sizeMap[hira] += 1
                else:
                    sizeMap[hira] = 1
                parentMap[hira_first] = root 
                parentMap[hira_second] = hira_first
                parentMap[hira] = hira_second 

        for id, parent in parentMap.items():
            tree.append([id, parent, sizeMap[id]])


        response_data = { 
            'tree': json.dumps(tree)
        }
                
        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )
    else:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_Type="application/json"
        )

class WordListView(ListView):
    model = Word

class GrammarGame(TemplateView):
    template_name = 'nlp/grammar_game.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context
        
# make ADP a varable and allow kanji / verb etc same game
def init_grammar_game(request):
    if request.method == 'POST':
        row = articles.sample(n=1).dropna(axis=1)
        nsents = row.columns.str.startswith('sent').sum()
        rand_col_num = random.randint(0, nsents - 1)
        goal = row[f'sent{rand_col_num}'].item()

        words = [ l for s in nlp(goal).sents for l in s ] 
        # [ print(_.pos_) for _ in words ]
        question = [ (_.text, False) if not _.pos_ == "ADP" else ('○', _.text) for _ in words ]
        parts = [ _.text for _ in words if _.pos_ == "ADP"]
        response_data = { 
            'question': question,
            'parts': parts,
            'goal': goal,
        }
                
        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )
    else:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_Type="application/json"
        )

class KanjiGame(TemplateView):
    template_name = 'nlp/kanji_game.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

def flatten_list(list):
    return [ l for s in list for l in s ]

def get_words():
    row = articles.sample(n=1).dropna(axis=1)
    article = row['Text'].item()
    nsents = row.columns.str.startswith('sent').sum()
    rand_col_num = random.randint(0, nsents - 1)
    goal = row[f'sent{rand_col_num}'].item()
    # sent_words = flatten_list(nlp(goal).sents) 
    sent_words = flatten_list(ginza.bunsetu_spans(nlp(goal))) 
    # [print(_.tag_) for _ in sent_words ]
    conv_words = flatten_list([ kks.convert(word.text) for word in sent_words if '普通名詞' in word.tag_ and not word.is_stop ])
    words = [ (_['orig'], _['hira']) for _ in conv_words if not _['orig'] == _['hira'] ]
    words = list(set(words))
    return (words, goal, article)

max_percent_old_words = 0.5
min_known_thresh = 2 
def words_worth_learning(words):
    if (len(words) == 0): return False
    already_known = []
    for word, hira in words:
        found_words = Word.objects.filter(word=word, hira=hira)
        if len(found_words) > 0:
            already_known += [ _ for _ in found_words if _.known > min_known_thresh ]
    print(len(already_known) / len(words))
    if (len(already_known) / len(words) < max_percent_old_words): return True
    return False

def init_kanji_game(request):
    if request.method == 'POST':
        words, text, article = [], '', ''
        ntries = 100
        while not words_worth_learning(words):
            ntries -= 1
            if ntries <= 0: break
            words, text, article = get_words()

        response_data = { 
            'article': article,
            'text': text,
            'words': words,
        }
                
        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )
    else:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_Type="application/json"
        )

def kanji_example(request):
    if request.method == 'POST':
        kanji = request.POST.get('kanji')
        sentences = []
        for index in range(40):
            clean_column = f'clean_sent{index}'
            sent_column = f'sent{index}'
            found_rows = articles[clean_column].apply(lambda x: (not isinstance(x, float)) and kanji in x.split())
            nfound = found_rows.sum()
            articles.loc[found_rows].sample(min(nfound, 1)).apply(
                lambda r: sentences.append(r[sent_column]),
                axis=1    
            )
            if len(sentences) > 0: break

        response_data = { 
            'sentences': sentences
        }
                
        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )
    else:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_Type="application/json"
        )

class WordReview(ListView):
    template_name = 'nlp/word_review.html'
    model = Word
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['object_list'] = sorted(Word.objects.all(), key=lambda _ : _.known)
        return context

def update_word_known(request):
    if request.method == 'POST':
        pk = request.POST.get('pk')
        op = request.POST.get('op')

        words = []
        if pk:
            words = Word.objects.filter(pk=pk)
        else:
            kanji = request.POST.get('kanji')
            hira = request.POST.get('hira')
            words = Word.objects.filter(kanji=kanji, hira=hira)        
       
        if (len(words) > 0):
            word = words[0]

            if (op == 'okay'):
                word.known = min(word.known * 2.5, 5000)
            elif (op == 'dame'):
                word.known = 1.0
            elif (op == 'known'):
                word.known = 3000.0

            word.save()

        response_data = { 
            'success': 'success'
        }
                
        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )
    else:
        return HttpResponse(
            json.dumps({"nothing to see": "this isn't happening"}),
            content_Type="application/json"
        )
