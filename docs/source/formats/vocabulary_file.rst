Format of the Vocabulary File
=============================

Vocabulary files assign ids to certain attribute values. We use standard text files to represent them.

Files are named in the following way:  

``<dataset>.vocabulary.<attribute>.txt``

Examples:

``ml-1m.vocabulary.title.txt, ml-1m.vocabulary.userId.txt,...``

The file content is displayed in tabular format:
 - Columns are seperated by tab.
 - The first column contains the attribute values and the second one has the associated ids.
 - Rows are sorted in ascending order by the second column.
 - The first few rows - and thus the first few ids - are always used for special tokens like padding, mask and unknown. 

Example:

``ml-1m.vocabulary.title.txt``::

    <PAD>	0
    <MASK>	1
    <UNK>	2
    Girl, Interrupted (1999)	3
    Back to the Future (1985)	4
    Titanic (1997)  5
    Cinderella (1950)   6
    Meet Joe Black (1998)	7


