## A Complete Japanese Learning Platform using Every NHK-Web-Easy article

### To run
Clone the repo and run `python manage.py runserver` from the top directory　

### home page
Enter a word and click submit. Related terms will be displayed as well as example sentences. Click an example sentence to view the full article containing that sentence and an analysis of the article. Click words in an article to save them as flash cards (saved words).  

<img src="doc-images/home.png?raw=true" width="360">

### text page (/text)
Enter Japanese text to have it analyzed into a summary, most common parts of speech and entities. Click words to save them.  

<img src="doc-images/vocab.png?raw=true" width="360">

### grammar game (/grammar_game)
An example sentence is shown with all particles (は、が、と etc) taken out. Drag the particles to their correct places to complete the game, or click solve to view the answer.  

<img src="doc-images/grammar_game.png?raw=true" width="360">

### kanji game (/kanji_game)
Drag each kanji to it's correct reading. Kanji are color coded by vowel sound. Click 'full article' to read the full article containing those kanji. Articles are chosen based on the users currently saved words and sorted by those which contain only one unknown word per sentence (i + 1).  

<img src="doc-images/kanji_game.png?raw=true" width="360">

### word review (/word_review)  
Displays flashcards for each saved words. Clicking a flashcard will show the reading for that word. Click わかった、オーケー、or ダメ to tell the site how well you can remember each word. Cards are sorted based on your understanding (spaced repetition)  

<img src="doc-images/word_review.png?raw=true" width="360">

### saved words (/saved)  
A list of all saved words. Click a word to show a tree graph of all sentences in NHK-Web-Easy containing that word at the bottom of the page. The graph can be used to view all uses of that word at the same time, to give a really good idea of the context surrounding each word. 

<img src="doc-images/word_tree.png?raw=true" width="360">

