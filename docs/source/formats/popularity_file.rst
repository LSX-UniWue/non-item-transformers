Format of the Popularity File
=============================

Popularity files assign the relative frequencies to attribute values. We use standard text files to represent them.

Files are named in the following way:  

``<dataset>.popularity.<attribute>.txt``

Examples:

``ml-1m.popularity.title.txt, ml-1m.popularity.userId.txt,...``

- The file content is displayed in one column. 
- The popularity value corresponds to the attribute value in the same row of the respective vocabulary file. 
- The first rows are zero-valued since they belong to special tokens (like PAD, MAS, UNK).
 

Example:

``ml-1m.popularity.age.txt``::

    0.0
    0.0
    0.0
    0.027205314089355323
    0.03877189667359522
    0.39547334607067125
    0.08361552435540973
    0.07247485275577405
    0.19896141706383366
    0.1834976489913608

