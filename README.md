A FOFE-based Local Detection Approach for Named Entity Recognition and Mention Detection
==========

This repository implements [A FOFE-based Local Detection Approach for Named Entity Reconigtion and Mention Detection](https://arxiv.org/abs/1611.00801) with [TensorFlow](https://www.tensorflow.org/). It ranks 2nd in the [Entity Discovery and Linking (EDL)](http://nlp.cs.rpi.edu/kbp/2016/) in [TAC Knowledge Base Population (KBP) 2016](https://tac.nist.gov//2016/KBP/).  



## This document intends to complement the data processing part that is not detailed in the paper. It does NOT explain the algorithms discussed in the paper. 



## LICENSE
Copyright (c) 2016 iNCML (Author: Mingbin Xu)

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE. 



## install.sh
Most scripts hard codes the paths to pre-trained word embeddings as well as training, development and test sets. ```install.sh``` organizes directories by establishing symbolic links. 



## skipgram-trainer.py & cmn-word-avg.py
A wrapper of [gensim](https://radimrehurek.com/gensim/) skip-gram. 

* When run on a Chinese dataset, the user needs to indicate whether word embedding or character embedding is desired. 
* When run on an English/Spanish dataset, the user needs to indicate whether the desired word embedding is case-sensitive or case-insensitive. 
* Words containing number are mapped to some predified labels, e.g. ```<date-value>```, ```<time-value>```, ```<phone-number>```, etc.

Because the detector is character-based in Chinese, each character is represented by the concatenation of character embedding and the word embedding where the character is from. Such word embedding is divided by the length of the number of characters. 



## setup.py & source/gigaword2feature.pyx
```source/gigaword2feature.pyx``` is written with [Cython](http://cython.org/). It is compiled by
```bash
python setup.py build_ext -i
```
and ```gigaword2feature.so``` will be generated in the root directory. 

Here are the basic ideas:

When a dataset is loaded, we pre-compute and cache the following for each sentence in the memory (Please pay more attention to the ```inclusive``` and ```exclusive``` index descriptions): 

* ```vector<int> numeric```:  
mapping from words (strings) to word-ids (integers).

* ```vector<string> sentence```:  
copy of the original sentence, required by character CNN.

* ```vector<vector<int>> left_context_idx``` and ```vector<vector<float>> left_context_data```:  
Let's say our vocabulory is of size 16 and we want to cache ```[0.243, 0, 0.49, 0, 0, 0, 0, 1, 0, 0, 0.7, 0, 0, 0, 0, 0]```. We cannot afford FOFE's dense form. However, FOFE is very sparse. We would like to record down indices of non-zero cells and the corresponding values, which leads to 2 vectors ```[0, 2, 7, 10]``` and ```[0.243, 0.49, 1, 0.7]```. ```left_contex_idx``` and ```left_context_data``` have the same length as ```numeric```. ```left_context_idx[i]``` and ```left_context_data[i]``` together represents the compressed left FOFE from the begining of the sentence to the ```ith``` word (inclusively).

* ```vector<vector<int>> right_context_idx``` and ```vector<vector<float>> left_context_data```:  
similar to left ones, ```right_context_idx[i]``` and ```right_context_data[i]``` together represents the compressed right FOFE from the end of the sentence to the ```ith``` word (inclusively).

With the above cached, constructing the representation of the phrase from the ```ith``` (inclusive) word to the ```jth``` (exclusive) word is fast and easy, conceptually: 

if excluding focus word(s)
```python
fofe_idx = numpy.concat(left_context_idx[i - 1], right_context_idx[j] + vocab_size)
fofe_data = numpy.concat(left_context_data[i - 1], right_context_data[j])
```

if including focus word(s)
```python
fofe_idx = numpy.concat(left_context_idx[j - 1], right_context_idx[i] + vocab_size)
fofe_data = numpy.concat(left_context_data[j - 1], right_context_data[i])
```

Each training example is associated with a sentence id, a starting index (inclusive) and an ending index (exclusive). A single-producer-single-consumer approach is applied for mini-batch preparation. 



## fofe_mention_net.py
FOFE & DNN models are defined. It defines the training behavior and the evaluatoin behavior. 


## kbp-xml-parser.py & iflytek-parser.py

Here's an example of LDC's file (Gigaword is also in the same format). In KBP, mention types and the corresponding character spanning are expected. The characters are zero-indexing. The offset of the first character (it must be '<') of the very first ```<doc>``` tag is 0.

```xml
<doc id="ENG_DF_000200_20061121_G00A06WU8">
<headline>
Random Variables Question
</headline>
<post author="Aralus" datetime="2006-11-21T13:12:00" id="p1">
1) Let X, Y and Z be independent random variables, each with mean mu and variance (sigma)^2. Find E(X + Y - 2Z)^2.

2) Let X be a random variable with p.d.f given by:

f(x) = 2x for 0 &lt;= x &lt;= 1

and f(x) = 0 otherwise. Use this example to the show that, in general,

E(1/X) is not equal to 1/(EX)

Any help would be appreciated.
</post>
<post author="James" datetime="2006-11-21T14:01:00" id="p2">
For (1):

Let A=X+Y-2Z
Now remember Var(A) = E(A^2)-E(A)^2

Are you trying to find E(A^2) or E(A)^2 ?
The latter is simple. The former can be found by calculating the variance and mean of A, and then using the above formula.

Remember:
E(aX+b)=aE(x)+b
Var(aX+b)=a^2 Var(X)
where X is a r.v. and a,b are real constants.
</post>
</doc>
```

The parser makes advantage of ```ssplit``` and ```tokenize``` from [CoreNLP Server](http://stanfordnlp.github.io/CoreNLP/corenlp-server.html). It removes all XML tags and assign beginning and ending character offsets to each word (inclusive-exclusive pairs). The results include the parsed content and a direct copy of the original file, separated by 128 "=". First part is dividied by 2 parts too. The first is the tokenized sentences while the second part is to extract "post author" from the xml tags. The above example leads the following:

#### if solution is not given (also input format of the evaluator)

```xml
Random Variables Question
(55,61) (62,71) (72,80)

1 -RRB- Let X , Y and Z be independent random variables , each with mean mu and variance -LRB- sigma -RRB- ^ 2 .
(155,156) (156,157) (158,161) (162,163) (163,164) (165,166) (167,170) (171,172) (173,175) (176,187) (188,194) (195,204) (204,205)
 (206,210) (211,215) (216,220) (221,223) (224,227) (228,236) (237,238) (238,243) (243,244) (244,245) (245,246) (246,247)

Find E -LRB- X Y - 2Z -RRB- ^ 2 .
(248,252) (253,254) (254,255) (255,256) (259,260) (261,262) (263,265) (265,266) (266,267) (267,268) (268,269)

2 -RRB- Let X be a random variable with p.d.f given by : f -LRB- x -RRB- = 2x for 0 < = x < = 1 and f -LRB- x -RRB- = 0 otherwise
 .
(271,272) (272,273) (274,277) (278,279) (280,282) (283,284) (285,291) (292,300) (301,305) (306,311) (312,317) (318,320) (320,321)
 (323,324) (324,325) (325,326) (326,327) (328,329) (330,332) (333,336) (337,338) (339,343) (343,344) (345,346) (347,351) (351,352
) (353,354) (356,359) (360,361) (361,362) (362,363) (363,364) (365,366) (367,368) (369,378) (378,379)

Use this example to the show that , in general , E -LRB- 1 X -RRB- is not equal to 1 -LRB- EX -RRB- Any help would be appreciated
 .
(380,383) (384,388) (389,396) (397,399) (400,403) (404,408) (409,413) (413,414) (415,417) (418,425) (425,426) (428,429) (429,430)
 (430,431) (432,433) (433,434) (435,437) (438,441) (442,447) (448,450) (451,452) (453,454) (454,456) (456,457) (459,462) (463,467
) (468,473) (474,476) (477,488) (488,489)

For -LRB- 1 -RRB- : Let A = X Y - 2Z Now remember Var -LRB- A -RRB- = E -LRB- A ^ 2 -RRB- - E -LRB- A -RRB- ^ 2 Are you trying to
 find E -LRB- A ^ 2 -RRB- or E -LRB- A -RRB- ^ 2 ?
(559,562) (563,564) (564,565) (565,566) (566,567) (569,572) (573,574) (574,575) (575,576) (577,578) (578,579) (579,581) (582,585)
 (586,594) (595,598) (598,599) (599,600) (600,601) (602,603) (604,605) (605,606) (606,607) (607,608) (608,609) (609,610) (610,611
) (611,612) (612,613) (613,614) (614,615) (615,616) (616,617) (619,622) (623,626) (627,633) (634,636) (637,641) (642,643) (643,64
4) (644,645) (645,646) (646,647) (647,648) (649,651) (652,653) (653,654) (654,655) (655,656) (656,657) (657,658) (659,660)

The latter is simple .
(661,664) (665,671) (672,674) (675,681) (681,682)

The former can be found by calculating the variance and mean of A , and then using the above formula .
(683,686) (687,693) (694,697) (698,700) (701,706) (707,709) (710,721) (722,725) (726,734) (735,738) (739,743) (744,746) (747,748)
 (748,749) (750,753) (754,758) (759,764) (765,768) (769,774) (775,782) (782,783)

Remember : E -LRB- aX b -RRB- = aE -LRB- x -RRB- b Var -LRB- aX b -RRB- = a ^ 2 Var -LRB- X -RRB- where X is a r.v. and a , b are
 real constants .
(785,793) (793,794) (795,796) (796,797) (797,799) (800,801) (801,802) (802,803) (803,805) (805,806) (806,807) (807,808) (809,810)
 (811,814) (814,815) (815,817) (818,819) (819,820) (820,821) (821,822) (822,823) (823,824) (825,828) (828,829) (829,830) (830,831
) (832,837) (838,839) (840,842) (843,844) (845,849) (850,853) (854,855) (855,856) (856,857) (858,861) (862,866) (867,876) (876,87
7)


(0,43)         <doc id="ENG_DF_000200_20061121_G00A06WU8">

(44,54)         <headline>

(81,92)         </headline>

(93,154)         <post author="Aralus" datetime="2006-11-21T13:12:00" id="p1">
(107,113) Aralus

(490,497)         </post>

(498,558)         <post author="James" datetime="2006-11-21T14:01:00" id="p2">
(512,517) James

(878,885)         </post>

(886,892)         </doc>



================================================================================================================================


<doc id="ENG_DF_000200_20061121_G00A06WU8">
<headline>
Random Variables Question
</headline>
<post author="Aralus" datetime="2006-11-21T13:12:00" id="p1">
1) Let X, Y and Z be independent random variables, each with mean mu and variance (sigma)^2. Find E(X + Y - 2Z)^2.

2) Let X be a random variable with p.d.f given by:

f(x) = 2x for 0 &lt;= x &lt;= 1

and f(x) = 0 otherwise. Use this example to the show that, in general,

E(1/X) is not equal to 1/(EX)

Any help would be appreciated.
</post>
<post author="James" datetime="2006-11-21T14:01:00" id="p2">
For (1):

Let A=X+Y-2Z
Now remember Var(A) = E(A^2)-E(A)^2

Are you trying to find E(A^2) or E(A)^2 ?
The latter is simple. The former can be found by calculating the variance and mean of A, and then using the above formula.

Remember:
E(aX+b)=aE(x)+b
Var(aX+b)=a^2 Var(X)
where X is a r.v. and a,b are real constants.
</post>
</doc>
```

#### if solution is given (also input/output format of the trainer/evaluator)

```xml
Random Variables Question
(55,61) (62,71) (72,80)

1 -RRB- Let X , Y and Z be independent random variables , each with mean mu and variance -LRB- sigma -RRB- ^ 2 .
(155,156) (156,157) (158,161) (162,163) (163,164) (165,166) (167,170) (171,172) (173,175) (176,187) (188,194) (195,204) (204,205)
 (206,210) (211,215) (216,220) (221,223) (224,227) (228,236) (237,238) (238,243) (243,244) (244,245) (245,246) (246,247)
(7,8,DUMMY,PER,NAM)

Find E -LRB- X Y - 2Z -RRB- ^ 2 .
(248,252) (253,254) (254,255) (255,256) (259,260) (261,262) (263,265) (265,266) (266,267) (267,268) (268,269)

2 -RRB- Let X be a random variable with p.d.f given by : f -LRB- x -RRB- = 2x for 0 < = x < = 1 and f -LRB- x -RRB- = 0 otherwise
 .
(271,272) (272,273) (274,277) (278,279) (280,282) (283,284) (285,291) (292,300) (301,305) (306,311) (312,317) (318,320) (320,321)
 (323,324) (324,325) (325,326) (326,327) (328,329) (330,332) (333,336) (337,338) (339,343) (343,344) (345,346) (347,351) (351,352
) (353,354) (356,359) (360,361) (361,362) (362,363) (363,364) (365,366) (367,368) (369,378) (378,379)

Use this example to the show that , in general , E -LRB- 1 X -RRB- is not equal to 1 -LRB- EX -RRB- Any help would be appreciated
 .
(380,383) (384,388) (389,396) (397,399) (400,403) (404,408) (409,413) (413,414) (415,417) (418,425) (425,426) (428,429) (429,430)
 (430,431) (432,433) (433,434) (435,437) (438,441) (442,447) (448,450) (451,452) (453,454) (454,456) (456,457) (459,462) (463,467
) (468,473) (474,476) (477,488) (488,489)

For -LRB- 1 -RRB- : Let A = X Y - 2Z Now remember Var -LRB- A -RRB- = E -LRB- A ^ 2 -RRB- - E -LRB- A -RRB- ^ 2 Are you trying to
 find E -LRB- A ^ 2 -RRB- or E -LRB- A -RRB- ^ 2 ?
(559,562) (563,564) (564,565) (565,566) (566,567) (569,572) (573,574) (574,575) (575,576) (577,578) (578,579) (579,581) (582,585)
 (586,594) (595,598) (598,599) (599,600) (600,601) (602,603) (604,605) (605,606) (606,607) (607,608) (608,609) (609,610) (610,611
) (611,612) (612,613) (613,614) (614,615) (615,616) (616,617) (619,622) (623,626) (627,633) (634,636) (637,641) (642,643) (643,64
4) (644,645) (645,646) (646,647) (647,648) (649,651) (652,653) (653,654) (654,655) (655,656) (656,657) (657,658) (659,660)

The latter is simple .
(661,664) (665,671) (672,674) (675,681) (681,682)

The former can be found by calculating the variance and mean of A , and then using the above formula .
(683,686) (687,693) (694,697) (698,700) (701,706) (707,709) (710,721) (722,725) (726,734) (735,738) (739,743) (744,746) (747,748)
 (748,749) (750,753) (754,758) (759,764) (765,768) (769,774) (775,782) (782,783)

Remember : E -LRB- aX b -RRB- = aE -LRB- x -RRB- b Var -LRB- aX b -RRB- = a ^ 2 Var -LRB- X -RRB- where X is a r.v. and a , b are
 real constants .
(785,793) (793,794) (795,796) (796,797) (797,799) (800,801) (801,802) (802,803) (803,805) (805,806) (806,807) (807,808) (809,810)
 (811,814) (814,815) (815,817) (818,819) (819,820) (820,821) (821,822) (822,823) (823,824) (825,828) (828,829) (829,830) (830,831
) (832,837) (838,839) (840,842) (843,844) (845,849) (850,853) (854,855) (855,856) (856,857) (858,861) (862,866) (867,876) (876,87
7)
(13,14,DUMMY,PER,NAM)


(0,43)         <doc id="ENG_DF_000200_20061121_G00A06WU8">

(44,54)         <headline>

(81,92)         </headline>

(93,154)         <post author="Aralus" datetime="2006-11-21T13:12:00" id="p1">
(107,113) Aralus

(490,497)         </post>

(498,558)         <post author="James" datetime="2006-11-21T14:01:00" id="p2">
(512,517) James

(878,885)         </post>

(886,892)         </doc>





================================================================================================================================


<doc id="ENG_DF_000200_20061121_G00A06WU8">
<headline>
Random Variables Question
</headline>
<post author="Aralus" datetime="2006-11-21T13:12:00" id="p1">
1) Let X, Y and Z be independent random variables, each with mean mu and variance (sigma)^2. Find E(X + Y - 2Z)^2.

2) Let X be a random variable with p.d.f given by:

f(x) = 2x for 0 &lt;= x &lt;= 1

and f(x) = 0 otherwise. Use this example to the show that, in general,

E(1/X) is not equal to 1/(EX)

Any help would be appreciated.
</post>
<post author="James" datetime="2006-11-21T14:01:00" id="p2">
For (1):

Let A=X+Y-2Z
Now remember Var(A) = E(A^2)-E(A)^2

Are you trying to find E(A^2) or E(A)^2 ?
The latter is simple. The former can be found by calculating the variance and mean of A, and then using the above formula.

Remember:
E(aX+b)=aE(x)+b
Var(aX+b)=a^2 Var(X)
where X is a r.v. and a,b are real constants.
</post>
</doc>
```

For example, the second sentence is (though incorrectly) labeled ```(7,8,DUMMY,PER,NAM)```. The label is incorrect because the given solution is correct. ```post author``` is automaticlly labeled as ```PERSON NAME```.



## kbp-system.py & kbp-ed-trainer.py & conll2003-ner-trainer.py
They are training schedulers that takes [parsed article with labels](#if-solution-is-given-also-inputoutput-format-of-the-trainerevaluator). 



## kbp-ed-evaluator & conll2003-ner-evaluator.py
They takes [parsed article without labels](#if-solution-is-not-given-also-input-format-of-the-evaluator) and produces [parsed article with labels](#if-solution-is-given-also-inputoutput-format-of-the-trainerevaluator).


## reformat.py & merge-tabs.py
They convert [parsed artiles with labels](#if-solution-is-given-also-inputoutput-format-of-the-trainerevaluator) back the format of KBP golden tab. 