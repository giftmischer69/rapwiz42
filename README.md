# [rapwiz42](https://pelgo14.github.io/artificial-rapper)

rapwiz42 is a neural network (gpt-2) and lyric corpus based lyric generation system.

first, given a list of names of artists, their lyrics are scraped from genius.com. 

```python
artists = ["aesop-rock", "mf-doom"] 

import lyricsgenius as genius

# scrape
api = genius.Genius(self.genius_key)
iter_artist = iter(self.artists)
for artist in iter_artist:
    artist_api = api.search_artist(artist)
    try:
        artist_api.save_lyrics(overwrite=True)
    except:
        logging.warn(f"Oops! Cant download: {artist}")
```

then a text corpus is created from the scraped lyrics, and the corpus is cleaned of any bad characters. 

```python
import fileinput
import re

def extract(self):
    ...
    pass

def clean_folder(self):
    ...
    pass

def clean_lyricdb(self):
    # [intro: eminem] <- needs to be removed
    for line in fileinput.input(r"lyricdb.txt", inplace=True):
        if not re.search(r"([^\w\d\s',<?>])", line):
            print(line, end="")
```

output:

```text
scraping...
Searching for songs by aesop-rock...

Searching for songs by mf-doom...

Changing artist name to 'Aesop Rock'
Changing artist name to 'MF DOOM'
Song 1: "None Shall Pass"
Song 2: "Daylight"
Song 1: "Doomsday"
Song 2: "Beef Rapp"
...
Song 225: "Marble Cake"
Song 204: "#chessboxin"
Done. Found 226 songs.
Wrote `Lyrics_AesopRock.json`
Done. Found 204 songs.
Wrote `Lyrics_MFDOOM.json`
Wrote `lyricdb.txt`
```

then a gpt-2 model gets fine-tuned on the text corpus. 

```python
import gpt-2-simple as gpt2

steps = 1000

file_name = "lyricdb.txt"
gpt2.download_gpt2(model_name="345M")
sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name='345M',
              steps=steps,
              restore_from='fresh',
              run_name='run1',
              print_every=25,
              sample_every=250,
              save_every=500,
              overwrite=True
              )
```

output:

```text
Fetching checkpoint: 1.05Mit [00:00, 248Mit/s]                                                      

training...

Fetching encoder.json: 1.05Mit [00:00, 108Mit/s]
Fetching hparams.json: 1.05Mit [00:00, 175Mit/s]
Fetching model.ckpt.data-00000-of-00001: 1.42Git [00:14, 98.4Mit/s]
Fetching model.ckpt.index: 1.05Mit [00:00, 219Mit/s]                                                
Fetching model.ckpt.meta: 1.05Mit [00:00, 66.0Mit/s]
Fetching vocab.bpe: 1.05Mit [00:00, 147Mit/s]
Loading checkpoint models/345M/model.ckpt
INFO:tensorflow:Restoring parameters from models/345M/model.ckpt

INFO:tensorflow:Restoring parameters from models/345M/model.ckpt
  0%|          | 0/1 [00:00<?, ?it/s]

Loading dataset...

100%|██████████| 1/1 [00:02<00:00,  2.61s/it]

dataset has 422429 tokens
Training...
[25 | 64.72] loss=4.76 avg=4.76
[50 | 119.89] loss=4.76 avg=4.76
[75 | 175.34] loss=4.41 avg=4.64
[100 | 230.99] loss=3.30 avg=4.30
[125 | 286.61] loss=4.69 avg=4.38
[150 | 342.33] loss=3.13 avg=4.17
[175 | 397.96] loss=4.79 avg=4.26
[200 | 453.61] loss=4.24 avg=4.26
[225 | 509.29] loss=3.14 avg=4.13
[250 | 564.97] loss=4.15 avg=4.13
Saving checkpoint/run1/model-250

```

then, with this gpt-2 model fine-tuned to rap flavour, rhyming lines in an "aabbaabb..." scheme are generated.

```python
# ----------------------------------------
# Generator class
# ----------------------------------------

class Generator:
    def __init__(self, sess, log_level="INFO"):
        ...
        pass

    def generate(self, line_count=16):
        """
        generates line_count amount of lines. every second line will rhyme with the previous
        (the last word of every second line will rhyme with the last word of the previous line)
        """
        lines = [self.gen_next_non_rhyme_line()]
        for i in progressbar.progressbar(range(1, line_count)):
            last_line = lines[i - 1]
            if i % 2 == 1:
                line = self.gen_next_rhyme_line(last_line)
            else:
                line = self.gen_next_non_rhyme_line()
            lines.append(line)
        return lines

    def gen_next_non_rhyme_line(self):
        """
        returns a line that suffices the is_good metric
        """
        good_lines = []
        tries_per_iteration = 100
        i = 0
        while len(good_lines) == 0:
            ...
            for line in temp_lines:
                if self.is_good(line):
                    good_lines.append(line)
        return random.choice(good_lines)

    def gen_next_rhyme_line(self, last_line):
        """
        returns a line that suffices the is_good metric and rhymes with last_line
        """
        good_lines = []
        tries_per_iteration = 100
        i = 0
        while len(good_lines) == 0:
            ...
            for line in temp_lines:
                this_last_word = line.split()[-1]
                if self.is_good(line) and self.rhymes(last_word, this_last_word):
                    good_lines.append(line)
        return random.choice(good_lines)


    def is_good(self, line):
        # defines a minimal metric for a "good" line
        # if line is empty or doesn't exist -> not good
        # if line is too short or too long -> not good
        # if there are (almost) no rhyme for the last word -> not good
        # if line is a multi-line line -> not good
        # if the line passed all checks it is good

        # if line is empty or doesn't exist -> not good
        if not line or len(line) == 0:
            return False

        # if line is too short or too long -> not good
        syl_count = syllables.estimate(line)
        if not 7 < syl_count < 18:
            return False

        # if there are (almost) no rhyme for the last word -> not good
        last_word = line.split()[-1]
        rhyme_list = pronouncing.rhymes(last_word)
        if not len(rhyme_list) > 20:
            return False

        # if line is a multi-line line -> not good
        if len(line.split("\n")) != 1:
            return False

        # if the line passed all checks it is good
        return True

    def rhymes(self, last_word, this_last_word):
        return this_last_word in pronouncing.rhymes(last_word)


# ----------------------------------------
# main
# ----------------------------------------

line_count = 16 

gen = Generator(log_level="INFO")
lines = gen.generate(line_count, sess)

```

output:

```text
INFO:root:generating lines...
100% (3 of 3) |##########################| Elapsed Time: 0:02:55 Time:  0:02:55
INFO:root:generating lines... done!

I'm looking forward to popping by again and bringing back hot
While I'm at it, I should mention I finally got
Daredevil has been a part of the Marvel comic book pant
I hate to abuse your patience, but I just can't
```
