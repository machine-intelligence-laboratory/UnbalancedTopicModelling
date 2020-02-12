import artm

import numpy as np
import pandas as pd
import functools
import operator
from collections import OrderedDict


def weigh_average(gamma, x, axis):
    '''
    Calculates average of x with weights gamma.
    '''
    if isinstance(axis, int):
        nom = np.sum(gamma * x, axis=axis)
        denom = np.sum(gamma, axis=axis)
        nom *= (denom != 0).astype(int)
        denom += (denom == 0).astype(int)
        return nom / denom

    elif isinstance(axis, list):
        nom = gamma * x
        denom = gamma
        for ax_ind, ax in enumerate(sorted(axis)):
            nom = np.sum(nom, axis=ax-ax_ind)
            denom = np.sum(denom, axis=ax-ax_ind)
        nom *= (denom != 0).astype(int)
        denom += (denom == 0).astype(int)
        return nom / denom

    else:
        raise ValueError


class ModelStatistics():
    def __init__(self, model):
        self.model = model
        self.phi = model.get_phi()

    def calculate_n(self, batch_vectorizer):
        '''
        Calculates all model counters from the batch vectorizer.
        '''
        self.theta = self.model.transform(batch_vectorizer)
        self.pwd = np.dot(self.phi.values, self.theta.values)
        self.nwd = np.zeros((self.phi.shape[0], self.theta.shape[1]))
        self.nwd = pd.DataFrame(self.nwd, self.phi.index.tolist(), self.theta.columns.tolist())

        doc2token = {}
        for batch_id in range(len(batch_vectorizer._batches_list)):
            batch_name = batch_vectorizer._batches_list[batch_id]._filename
            batch = artm.messages.Batch()
            with open(batch_name, "rb") as f:
                batch.ParseFromString(f.read())

            for item_id in range(len(batch.item)):
                item = batch.item[item_id]
                theta_item_id = getattr(item, self.model.theta_columns_naming)

                doc2token[theta_item_id] = {'tokens': [], 'weights': []}
                for token_id, token_weight in zip(item.token_id, item.token_weight):
                    doc2token[theta_item_id]['tokens'].append(batch.token[token_id])
                    doc2token[theta_item_id]['weights'].append(token_weight)

                    self.nwd.loc[batch.token[token_id], theta_item_id] += token_weight

        previous_num_document_passes = self.model._num_document_passes
        self.model._num_document_passes = 10
        self.ptdw = self.model.transform(batch_vectorizer=batch_vectorizer, theta_matrix_type='dense_ptdw')
        self.model._num_document_passes = previous_num_document_passes

        docs = self.ptdw.columns
        docs_unique = OrderedDict.fromkeys(docs).keys()

        tokens = [doc2token[doc_id]['tokens'] for doc_id in docs_unique]
        tokens = functools.reduce(operator.iconcat, tokens, [])

        ndw = np.concatenate([np.array(doc2token[doc_id]['weights']) for doc_id in docs_unique])
        self._ndw = np.tile(ndw, (self.ptdw.shape[0], 1))

        self.ptdw.columns = pd.MultiIndex.from_arrays([docs, tokens], names=('doc', 'token'))
        self.ntdw = self.ptdw * self._ndw

        self.ntd = self.ntdw.groupby(level=0, axis=1).sum()
        self.nwt = self.ntdw.groupby(level=1, axis=1).sum().T
        self.nwd = self.nwd.values
        self.nt = self.nwt.sum(axis=0).values
        self.nd = self.ntd.sum(axis=0).values

    def recalculate_n(self, batch_vectorizer):
        self.theta = self.model.transform(batch_vectorizer)
        self.pwd = np.dot(self.phi.values, self.theta.values)

        previous_num_document_passes = self.model._num_document_passes
        self.model._num_document_passes = 10
        self.ptdw = pd.DataFrame(
            self.model.transform(batch_vectorizer=batch_vectorizer, theta_matrix_type='dense_ptdw').values,
            self.ptdw.index,
            self.ptdw.columns
        )
        self.model._num_document_passes = previous_num_document_passes

        self.ntdw = self.ptdw * self._ndw

        self.ntd = self.ntdw.groupby(level=0, axis=1).sum()
        self.nwt = self.ntdw.groupby(level=1, axis=1).sum().T
        self.nt = self.nwt.sum(axis=0).values
        self.nd = self.ntd.sum(axis=0).values

    def calculate_s_t(self, batch_vectorizer, alpha=None, use_ptdw=None, calculate_n=False):
        '''
        Calculates semantic heterogenity of topic in model.
        '''
        if calculate_n:
            self.calculate_n(batch_vectorizer)
        
        if alpha is not None:
            model_loss = (self.pwd < alpha / self.nd).astype(int)
        else:
            model_loss = np.log(self.nwd / self.nd / self.pwd + (self.nwd == 0).astype(int))

        s_t = np.zeros(self.ntd.shape[0])
        for t in range(s_t.shape[0]):
            ptwd_t = np.matmul(self.phi.iloc[:, t].values.reshape(-1, 1), self.theta.iloc[t, :].values.reshape(1, -1))
            if use_ptdw:
                s_t[t] = weigh_average(ptwd_t, model_loss, [0,1])
            else:
                ntwd_t = self.nwd * ptwd_t / self.pwd
                s_t[t] = weigh_average(ntwd_t, model_loss, [0,1])
        return s_t

    def calculate_imp_t(self, batch_vectorizer, binary_loss=False, use_ptdw=False, calculate_n=False):
        '''
        Calculates topic impurity for every topic in model.
        '''
        if calculate_n:
            self.calculate_n(batch_vectorizer)
        
        imp_t = np.zeros(self.ntd.shape[0])
        for t in range(imp_t.shape[0]):
            if binary_loss:
                model_loss = self.phi.values[:, :] > self.phi.values[:, t][:, np.newaxis]
                model_loss = np.sum(model_loss, axis=1).astype(bool).astype(int)
            else:
                phi_safe = self.phi.values[:, t][:, np.newaxis] + (self.phi.values[:, t][:, np.newaxis] == 0).astype(int)
                model_loss = np.log(self.phi.values / phi_safe)
                idxmax = np.argmax(model_loss, axis=1)
                model_loss = model_loss[np.arange(len(model_loss)), idxmax]

            ptwd_t = np.matmul(self.phi.iloc[:, t].values.reshape(-1, 1), self.theta.iloc[t, :].values.reshape(1, -1))
            if use_ptdw:
                imp_t[t] = weigh_average(ptwd_t, model_loss[:, np.newaxis], [0,1])
            else:
                ntwd_t = self.nwd * ptwd_t / self.pwd
                imp_t[t] = weigh_average(ntwd_t, model_loss[:, np.newaxis], [0,1])
        return imp_t

    def calculate_s_td(self, batch_vectorizer, alpha=None, use_ptdw=False, calculate_n=False):
        '''
        Calculates document coherence with topic.
        '''
        if calculate_n:
            self.calculate_n(batch_vectorizer)

        if alpha is not None:
            model_loss = (self.pwd < alpha / self.nd).astype(int)
        else:
            model_loss = np.log(self.nwd / self.nd / self.pwd + (self.nwd == 0).astype(int))

        s_td = np.zeros(self.ntd.shape)
        for t in range(s_td.shape[0]):
            ptwd_t = np.matmul(self.phi.iloc[:, t].values.reshape(-1, 1), self.theta.iloc[t, :].values.reshape(1, -1))
            if use_ptdw:
                s_td[t, :] = weigh_average(ptwd_t, model_loss, 0)
            else:
                ntwd_t = self.nwd * ptwd_t / self.pwd
                s_td[t, :] = weigh_average(ntwd_t, model_loss, 0)
        return s_td

    def calculate_s_wt(self, batch_vectorizer, alpha=None, use_ptdw=False, calculate_n=False):
        '''
        Calculates token coherence with topic.
        '''
        if calculate_n:
            self.calculate_n(batch_vectorizer)

        if alpha is not None:
            model_loss = (self.pwd < alpha / self.nd).astype(int)
        else:
            model_loss = np.log(self.nwd / self.nd / self.pwd + (self.nwd == 0).astype(int))

        s_wt = np.zeros(self.nwt.shape)
        for t in range(s_wt.shape[1]):
            ptwd_t = np.matmul(self.phi.iloc[:, t].values.reshape(-1, 1), self.theta.iloc[t, :].values.reshape(1, -1))
            if use_ptdw:
                s_wt[:, t] = weigh_average(ptwd_t, model_loss, 1)
            else:
                ntwd_t = self.nwd * ptwd_t / self.pwd
                s_wt[:, t] = weigh_average(ntwd_t, model_loss, 1)

        return s_wt

    def calculate_topic_statistics(self, batch_vectorizer, alpha=1, recalculate_n=True, calculate_n=False):
        '''
        Calculates topic semantic heterogenity and topic impurity
        with likelihood loss and binary loss,
        with and without tolerance to the word burstiness.
        '''
        if recalculate_n:
            self.recalculate_n(batch_vectorizer)
        if calculate_n:
            self.calculate_n(batch_vectorizer)

        s_t = self.calculate_s_t(batch_vectorizer)
        bin_s_t = self.calculate_s_t(batch_vectorizer, alpha=alpha)
        ptdw_s_t = self.calculate_s_t(batch_vectorizer, use_ptdw=True)
        bin_ptdw_s_t = self.calculate_s_t(batch_vectorizer, alpha=alpha, use_ptdw=True)

        imp_t = self.calculate_imp_t(batch_vectorizer)
        bin_imp_t = self.calculate_imp_t(batch_vectorizer, binary_loss=True)
        ptdw_imp_t = self.calculate_imp_t(batch_vectorizer, use_ptdw=True)
        bin_ptdw_imp_t = self.calculate_imp_t(batch_vectorizer, binary_loss=True, use_ptdw=True)

        return s_t, bin_s_t, ptdw_s_t, bin_ptdw_s_t, imp_t, bin_imp_t, ptdw_imp_t, bin_ptdw_imp_t
