# Neural Citation Network
Source code for SIGIR 17 paper:

Travis Ebesu, Yi Fang. Neural Citation Network for Context-Aware Citation Recommendation. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2017. [PDF](http://www.cse.scu.edu/~yfang/NCN.pdf)


```
@inproceedings{Ebesu:2017:NCN:3077136.3080730,
 author = {Ebesu, Travis and Fang, Yi},
 title = {Neural Citation Network for Context-Aware Citation Recommendation},
 booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
 series = {SIGIR '17},
 year = {2017},
 isbn = {978-1-4503-5022-8},
 location = {Shinjuku, Tokyo, Japan},
 pages = {1093--1096},
 numpages = {4},
 url = {http://doi.acm.org/10.1145/3077136.3080730},
 doi = {10.1145/3077136.3080730},
 acmid = {3080730},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {citation recommendation, deep learning, neural machine translation},
} 
```

# Dataset:
The Raw Refseer dataset can be downloaded from here:

https://psu.app.box.com/v/refseer

Preprocess the dataset according to the readme then remove the citations, math notation and lemmatize tokens. Take the top 20K vocabulary for the encoder citation contexts, 20K vocab for the decoder and 20K authors. The authors were determined by simple heuristic of first initial and last name. Once the data is formatted you may need to do a little changing to the code to feed in your data. 

The preprocessed dataset can be found here: 

https://drive.google.com/file/d/11dITvLfmCCJKhmXoPnTIL7A8wd1DBDHJ/view?usp=sharing


I used trec_eval for evaluating the results.

Note this requires TensorFlow r0.11


Please open a issue if you have any questions/problems I will try my best to assist you.
