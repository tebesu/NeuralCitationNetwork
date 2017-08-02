#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author:   Travis A. Ebesu
'''
# pylint: disable=line-too-long, too-few-public-methods, too-many-instance-attributes, redefined-outer-name
import atexit
import h5py
import numpy as np
import cPickle
from os.path import isfile


CITE_TOKEN = b"<CITE>"
GO_TOKEN = b"<GO>"
END_TOKEN = b"<EOS>"
PAD_TOKEN = b"<PAD>"
UNK_TOKEN = b"<UNK>"
DIGIT_TOKEN = b"<DIGIT>"

class HDF5Dataset(object):
    """
    Create a simple dataset, in which we can easily access.
    """

    def __init__(self, filename, write=False, compression_level=8,
                 write_mode='x', driver='core'):
        """
        HDF5 file wrapper to read/write objects via numpy arrays, protobufs,
        pickles.

        :param filename: string
        :param write: bool, if writing to file; if False and driver='core' we
                      load the entire file into memory.
        :param compression_level: 0-9, compression level
        :param write_mode: default x, a = append
        :param driver: string, set to None or core; use core for in memory
        """

        # Cache keys
        self._keys = None

        # Compression Level
        self._comp_level = compression_level
        self._meta_data = {}

        # When reading, load into memory
        if not write:
            if not isfile(filename): raise Exception("File %s does not exist" % filename)
            # Core driver, loads the entire dataset into memory
            self._file = h5py.File(filename, mode='r',
                                   libver='latest', driver=driver)
                                   #backing_store=False)
        else:
            # Normal exception thrown is not descriptive
            if write_mode in ['x', 'w-'] and isfile(filename):
                raise Exception('File %s already exists; either change write_mode or filename' % filename)

            self._file = h5py.File(filename, mode=write_mode,
                                   libver='latest', driver=driver)


    def close(self):
        self._file.close()


    def keys(self):
        # Cache attribute keys
        if self._keys:
            return self._keys
        self._keys = self._file.attrs.keys()
        return self._keys


    def _checkKeyStr(self, key):
        """
        Keys for h5py must be string/unicode type.
        This function checks it and converts if necessary.
        """
        if not (isinstance(key, str) or isinstance(key, unicode)):
            return unicode(key)
        return key

    def _checkKeys(self, key1, key2):
        """
        Check a single or two keys if they are a string or not
        if not convert them and return it
        """
        return self._checkKeyStr(key1), self._checkKeyStr(key2)

    def _checkDataset(self, group):
        self._requireObjDataset(group)

    def _requireObjDataset(self, group):
        """
        Create a base dataset we require, ie this is required when we
        add pickle, protobuf or dictionary
        """
        self._file.require_dataset(group, shape=(1,), shuffle=True, dtype=np.int8,
                              compression='gzip', compression_opts=self._comp_level)

    def addProtobuf(self, group, key, protobuf):
        self.addString(group, key, protobuf.SerializeToString())

    def addString(self, group, key, string):
        group, key = self._checkKeys(group, key)
        self._checkDataset(group)
        # We need to store as a numpy void
        self._file[group].attrs[key] = np.void(b"%s" % string)

    def addDict(self, group, key, dict_data):
        group, key = self._checkKeys(group, key)
        self._checkDataset(group)
        for k, v in dict_data.iteritems():
            self._file[group][key].attrs[k] = v

    def addPickle(self, group, key, obj):
        group, key = self._checkKeys(group, key)
        self._checkDataset(group)
        self._file[group].attrs[key] = np.void(b"%s" % cPickle.dumps(obj, cPickle.HIGHEST_PROTOCOL))

    def addArray(self, key, data):
        key = self._checkKeyStr(key)
        self._file.create_dataset(key, data=data,
                                  compression='gzip', compression_opts=self._comp_level)

    def getString(self, group, key):
        group, key = self._checkKeys(group, key)
        return self._file[group].attrs[key].tostring()

    def getProtobuf(self, group, key, pbuff_cls):
        return pbuff_cls.FromString(self.getString(self.getString(group, key)))

    def getDict(self, group, key):
        group, key = self._checkKeys(group, key)
        return {k: v for k, v in self._file[group][key].attrs.iteritems()}

    def getPickle(self, group, key):
        group, key = self._checkKeys(group, key)
        return cPickle.loads(self._file[group].attrs[key].tostring())

    def getArray(self, key):
        key = self._checkKeyStr(key)
        return self._file[key].value

    def __getitem__(self, key):
        return self._file[key]

    def __setset__(self, key, value):
        self._file[key] = value


class CitationContextDataset(HDF5Dataset):
    """
    H5Py
    |---- Vocabulary
          |= Encoder Vocab
          |= Decoder Vocab
    |
    |---- context[context_id]
          |= Encoder Sequences
          |= Decoder Sequences
    |
    |---- index
          |= Train keys
          |= Test keys
          |= Valid keys



    contexts = {
        'context': [0, 1, 2],
        'title': [0, 1],
    }

    self.addContext(context_id, contexts)

    papers[cluster_id] = {title: ..., authors: ...}

    context_sets[context_id] = [{ 'cited': 1, 'citing': 2}, ... ]

    context_maps[context_id] = {
        'cited': 0,
        'citing': [id1, id2]
    }
    """

    _KEY_TRAIN_INDEX = "indexTrain"
    _KEY_TEST_INDEX = "indexTest"
    _KEY_VALID_INDEX = "indexValid"
    # ---------------------------

    def addAuthorVocab(self, vocab_obj):
        self.addPickle('author', 'data', vocab_obj)

    def getAuthorVocab(self):
        return self.getPickle('author', 'data')

    # ---------------------------

    def addEncoderVocab(self, vocab_obj):
        self.addPickle('encoder', 'data', vocab_obj)

    def getEncoderVocab(self):
        return self.getPickle('encoder', 'data')

    # ---------------------------

    def addDecoderVocab(self, vocab_obj):
        self.addPickle('decoder', 'data', vocab_obj)

    def getDecoderVocab(self):
        return self.getPickle('decoder', 'data')

    # ---------------------------

    def addTrainIndex(self, data):
        self.addArray(self._KEY_TRAIN_INDEX, data)

    def getTrainIndex(self):
        return self.getArray(self._KEY_TRAIN_INDEX)

    def addTestIndex(self, data):
        self.addArray(self._KEY_TEST_INDEX, data)

    def getTestIndex(self):
        return self.getArray(self._KEY_TEST_INDEX)

    def addValidIndex(self, data):
        self.addArray(self._KEY_VALID_INDEX, data)

    def getValidIndex(self):
        return self.getArray(self._KEY_VALID_INDEX)

    def addYearClusters(data):
        self.addPickle("years", 0, data)

    def getYearClusters():
        return self.getPickle('years', 0)

    # ---------------------------

    def addContext(self, index, data):
        self.addPickle('context', index, data)

    def getContext(self, index):
        return self.getPickle('context', index)

    #     def addPaper(self, cluster_id, data):
    #         self.addPickle('paper', cluster_id, data)

    #     def getPaper(self, cluster_id):
    #         return self.getPickle('paper', cluster_id)

    # ---------------------------

    def addContextMaps(self, data):
        self.addPickle('maps', 'contexts', data)

    def getContextMaps(self):
        return self.getPickle('maps', 'contexts')

    def addContextSets(self, data):
        self.addPickle('maps', 'contextSets', data)

    def getContextSets(self):
        return self.getPickle('maps', 'contextSets')

    def addPaperClusters(self, data):
        self.addPickle('papers', 'clusters', data)

    def getPaperClusters(self):
        return self.getPickle('papers', 'clusters')

if __name__ == '__main__':
    pass
