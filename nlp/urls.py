from django.urls import path

from . import views

urlpatterns = [
    path('', views.HomePageView.as_view(), name='index'),
    path('vocab/', views.VocabView.as_view(), name='vocab'),
    path('text/', views.TextPageView.as_view(), name='text'),
    path('saved/', views.WordListView.as_view(), name='saved'),
    path('grammar_game/', views.GrammarGame.as_view(), name='grammar_game'),
    path('kanji_game/', views.KanjiGame.as_view(), name='kanji_game'),
    path('word_review/', views.WordReview.as_view(), name='word_review'),
    path('update/word/known', views.update_word_known, name='update_word_known'),
    path('ajax/get_vocab/', views.get_vocab, name='get_vocab'),
    path('ajax/input_text/', views.input_text, name='input_text'),
    path('ajax/input_match/', views.input_match, name='input_match'),
    path('ajax/add_word/', views.add_word, name='add_word'),
    path('ajax/make_tree/', views.make_tree, name='make_tree'),
    path('ajax/init_grammar_game/', views.init_grammar_game, name='init_grammar_game'),
    path('ajax/init_kanji_game/', views.init_kanji_game, name='init_kanji_game'),
    path('ajax/kanji_example/', views.kanji_example, name='kanji_example'),
]
