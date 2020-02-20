import numpy as np
import warnings


def get_top_values(values, top_number, threshold=None):
    if top_number > len(values):
        top_number = len(values)
        warnings.warn('num_top_tokens greater than modality size', UserWarning)

    top_indexes = np.argpartition(
        values, len(values) - top_number
    )[-top_number:]

    top_values = values[top_indexes]
    sorted_top_values_indexes = top_values.argsort()[::-1]

    return_indexes = top_indexes[sorted_top_values_indexes]
    return return_indexes, values[return_indexes]


def compute_blei_scores(phi):
    """
    Computes Blei score  
    phi[wt] * [log(phi[wt]) - 1/T sum_k log(phi[wk])]

    """  # noqa: W291
    topic_number = phi.shape[0]
    blei_eps = 1e-42
    log_phi = np.log(phi + blei_eps)
    denominator = np.sum(log_phi, axis=0)
    denominator = denominator[np.newaxis, :]

    if hasattr(log_phi, "values"):
        multiplier = log_phi.values - denominator / topic_number
    else:
        multiplier = log_phi - denominator / topic_number

    score = (phi * multiplier).transpose()
    return score


class BleiTopTokens():
    def __init__(self, num_top_tokens=10, ind2tok=None):
        self.num_top_tokens = num_top_tokens
        self.ind2tok = ind2tok

    def view(self, phi, threshold=None):
        target_values = compute_blei_scores(phi)

        phi = target_values.T
        topics_names = phi.columns

        topic_top_tokens = {}

        for topic_name in topics_names:
            topic_top_tokens[topic_name] = {}
            topic_column = phi[topic_name]

            top_tokens_indexes, top_tokens_values = get_top_values(topic_column.values,
                                                                   top_number=self.num_top_tokens,
                                                                   threshold=threshold)
            
            topic_top_tokens[topic_name]['values'] = top_tokens_values
            
            if self.ind2tok is not None:
                topic_top_tokens[topic_name]['tokens'] = list(map(lambda x: self.ind2tok[x], 
                                                                topic_column.index[top_tokens_indexes]))
            else:
                topic_top_tokens[topic_name]['tokens'] = topic_column.index[top_tokens_indexes].tolist()

        return topic_top_tokens
