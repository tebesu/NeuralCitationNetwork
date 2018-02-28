# Neural Citation Network
Source code for SIGIR 17 paper:

Travis Ebesu, Yi Fang. Neural Citation Network for Context-Aware Citation Recommendation. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2017. [PDF](http://www.cse.scu.edu/~yfang/NCN.pdf)


# Dataset:
The Raw Refseer dataset can be downloaded from here:

https://psu.app.box.com/v/refseer

Preprocess the dataset according to the readme then remove the citations, math notation and lemmatize tokens. Take the top 20K vocabulary for the encoder citation contexts, 20K vocab for the decoder and 20K authors. The authors were determined by simple heuristic of first initial and last name. Once the data is formatted you may need to do a little changing to the code to feed in your data. 

The preprocessed dataset can be found here:
https://drive.google.com/open?id=10tdKRxsq7v9-HdjUaFeqmrojKz6gpvcw


I used trec_eval for evaluating the results.

Note this requires TensorFlow r0.11


Please open a issue if you have any questions/problems I will try my best to assist you.
