import numpy as np
import pandas as pd
import warnings


class BaseRegularizer:
    """
    Base regularizer class to construct custom regularizers.
    """
    def __init__(self, name, tau, gamma=None):
        self.name = name
        self.tau = tau
        self.gamma = gamma
        self._model = None

    def attach(self, model):
        self._model = model

    def grad(self, pwt, nwt):
        raise NotImplementedError('grad method should be overrided in an inherited class')


class TopicPriorRegularizer(BaseRegularizer):
    """
    TopicPriorRegularizer adds prior beta_t to every column
    in Phi matrix of ARTM model. Thus every phi_wt has
    preassigned prior probability of being attached to topic t.

    If beta is balanced with respect to apriori collection balance,
    topics become better and save n_t balance.

    """  # noqa: W291
    def __init__(self, name, tau, num_topics=None, beta=1):
        super().__init__(name, tau)

        beta_is_n_dim = isinstance(beta, (list, np.ndarray))
        if beta_is_n_dim and (num_topics is not None) and len(beta) != num_topics:
            raise ValueError('Beta dimension doesn\'t equal num_topics.')
        if num_topics is None and not beta_is_n_dim:
            warnings.warn('Num topics set to 1.')
            num_topics = 1

        if beta_is_n_dim:
            if np.sum(np.array(beta)) == 0:
                raise ValueError('Incorrect input beta: at least one value must be greater zero.')
            if np.min(np.array(beta)) < 0:
                raise ValueError('Incorrect input beta: all values must be greater or equal zero.')

            self.beta = np.array(beta)
            self.beta = self.beta / np.sum(self.beta)
        else:
            self.beta = np.ones(num_topics)

    def grad(self, pwt, nwt):
        grad_array = np.repeat([self.beta * self.tau], pwt.shape[0], axis=0)

        return grad_array


class SemanticHeterogenityRegularizer(BaseRegularizer):
    """
    SemanticHeterogenityRegularizer aims to minimizing heterogenity of the topics.

    """  # noqa: W291
    def __init__(self, name, tau, semantic_statistics, batch_vectorizer):
        super().__init__(name, tau)

        self.statistics = semantic_statistics
        self.batch_vectorizer = batch_vectorizer
        
        self.statistics.calculate_n(self.batch_vectorizer)

    def grad(self, pwt, nwt):
        self.statistics.recalculate_n(self.batch_vectorizer)

        p_t = np.zeros(self.statistics.theta.shape[0])
        for t in range(p_t.shape[0]):
            ptwd_t = np.matmul(
                self.statistics.phi.iloc[:, t].values.reshape(-1, 1), 
                self.statistics.theta.iloc[t, :].values.reshape(1, -1)
            )
            p_t[t] = ptwd_t.sum()
        p_t = p_t / p_t.sum()

        gamma_wd = np.zeros(self.statistics.nwd.shape)
        for t in range(p_t.shape[0]):
            ptwd_t = np.matmul(
                self.statistics.phi.iloc[:, t].values.reshape(-1, 1), 
                self.statistics.theta.iloc[t, :].values.reshape(1, -1)
            )
            gamma_wd += ptwd_t / p_t[t]

        grad_array = np.zeros(self.statistics.nwt.shape)
        for t in range(p_t.shape[0]):
            grad_array[:, t] = np.sum(
                (self.statistics.theta.iloc[t, :].values.reshape(1, -1) * gamma_wd *
                 self.statistics.nwd) / self.statistics.pwd,
                axis=1
            )

        return grad_array


def custom_fit_offline(model, custom_regularizers, batch_vectorizer, num_collection_passes):
    for regularizer in custom_regularizers:
        regularizer.attach(model)
    
    base_regularizers_name = [regularizer.name for regularizer in model.regularizers.data.values()]
    base_regularizers_tau = [regularizer.tau for regularizer in model.regularizers.data.values()]
    
    for i in range(num_collection_passes):
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)
        
        pwt = model.get_phi(model_name=model.model_pwt) 
        nwt = model.get_phi(model_name=model.model_nwt) 
        
        rwt_name = 'rwt'
        
        model.master.regularize_model(pwt=model.model_pwt,
                                      nwt=model.model_nwt,
                                      rwt=rwt_name,
                                      regularizer_name=base_regularizers_name,
                                      regularizer_tau=base_regularizers_tau)
        (meta, nd_array) = model.master.attach_model('rwt')
        attached_rwt = pd.DataFrame(data=nd_array, columns=meta.topic_name, index=meta.token)
        
        for regularizer in custom_regularizers:
            attached_rwt.values[:, :] += regularizer.grad(pwt, nwt)
        
        model.master.normalize_model(pwt=model.model_pwt,
                                     nwt=model.model_nwt,
                                     rwt=rwt_name)