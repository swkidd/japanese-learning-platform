import spacy

nlp = spacy.load('ja_ginza')

def tree_traverse(root, func):
    if not root.text is None and len([*root.lefts]) == 0 and len([*root.rights]) == 0:
        func(root)
    else:
        [ tree_traverse(_, func) for _ in root.lefts]
        if not root.text is None:
            func(root)
        [ tree_traverse(_, func) for _ in root.rights]

s = '''むかしむかし、カツオ取りの漁師たちが遠くの海へ出かけましたが、目的の場所へ到着する前に夜になってしまいました。
　帰ろうにも向かい風が強くて、船が思う様に進みません。
「ああー、たいくつだな」
　見張りの若い男がぼんやりと海を眺めていると、向かい風に逆らいながら近づいて来る船がありました。
「おや、あれはなんだ？」
　その船は、船べりにも、ほづなにも、青白い炎が数え切れないほどともっています。
「ゆっ、幽霊船だ！」
　それは万灯船(まんとうせん)と呼ばれる幽霊船で、この辺りの海にだけ現れるのです。
　年配の漁師が、見張りの若い男に言いました。
「いいか。幽霊とは絶対に、口をきいてはいかんぞ」
「う、うん」
「それに、『ひしゃくで水をくれ』と言われたら、ひしゃくの底を抜いて渡すんだ。うっかり普通のひしゃくを渡したら、そのひしゃくで船に水を入れられて、船を沈められてしまうからな」
「わかった。口はきかずに、ひしゃくを渡す時は底を抜くんだな」

　やがて幽霊船は風に逆らいながらも滑る様に近づいて来て、漁船とへさきを並べました。
　船べりには、ひたいに三角のきれをつけた幽霊たちがいて、
「水をくれ～」
「頼むから、真水を飲ませてくれ～」
と、かぼそい声をしぼり出して言います。
　幽霊は、男だけではありません。
　女や子どもたちも、まじっています。
　これを見た船頭が、漁師たちに言いつけました。
「おい。水のたるを五つ六つ、持って来い」
「何を言うんだ！　そんなのとんでもねえ！」
　漁師たちは、反対しましたが、
「いいか。
　海の上では飲み水がないくらい、つらい事はない。
　相手が幽霊船だとしても、ここはなさけをかけてやろうではないか」
と、船頭は言って、幽霊船になわを投げ渡して水のたるを次々とつるし、幽霊たちにたぐらせました。
　船べりの幽霊たちは、うれしそうに水だるを受け取ると、ゆっくりとその場を離れていきました。

　やがて風もおさまって、朝にはすっかり波のおだやかな海になりました。
　そして漁を始めたところ、たちまちの大漁です。

　それからというもの、この船頭の船は漁に出るたびに、必ず大漁だったそうです。

おしまい'''

doc = nlp(s)
depSet = set()
[ tree_traverse(sent.root, lambda _: depSet.add(spacy.explain(_.dep_)) if _.dep_ is not None else None ) for sent in doc.sents ]

for sent in doc.sents:
    for child in sent.root.children:
        l = []
        tree_traverse(child, lambda _: l.append(_.text))
        print(child.dep_, ''.join(l))