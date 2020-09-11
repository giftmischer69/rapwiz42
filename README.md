# [rapwiz42](https://pelgo14.github.io/artificial-rapper)

[**rapwiz42 colab notebook**](https://colab.research.google.com/drive/1C4_CBrSJcUfRopQxaQqlrnU9Ve5Xk33F?usp=sharing)

rapwiz42 is a neural network (gpt-2) and lyric corpus based lyric generation system.

* first, given a list of names of artists, their lyrics are scraped from genius.com. 

* then a text corpus is created from the scraped lyrics, and the corpus is cleaned of any bad characters. 

* then a gpt-2 model gets fine-tuned on the text corpus. 

* then, with this gpt-2 model fine-tuned to rap flavour, rhyming lines in an "aabbaabb..." scheme are generated.
