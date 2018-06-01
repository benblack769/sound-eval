# sound-eval


https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer

Dataset location:

https://github.com/mdeff/fma

citation:

@inproceedings{fma_dataset,
  title = {FMA: A Dataset for Music Analysis},
  author = {Defferrard, Micha\"el and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  booktitle = {18th International Society for Music Information Retrieval Conference},
  year = {2017},
  url = {https://arxiv.org/abs/1612.01840},
}

* input: music database
* output: representation vectors for each song
* checks for accuracy:
	* Get hand classification of songs, see if vector can correctly identify classification
* cool extensions:
	* Does it improve prediction?
	* can this be used for compression?


Prerequisites:

1. Find large corpus of free music
2. Find some corpus of classified music (does not have to be the same as large corpus)
3. Build way of measuring classification success of vector rep
4. Build stub method that generates random vectors, show that it does not classify music correctly.
