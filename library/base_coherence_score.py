from collections import defaultdict
from enum import Enum, IntEnum, auto
import numpy as np
import pandas as pd
from typing import List, Dict

from topicnet.cooking_machine.models.base_model import BaseModel
from topicnet.cooking_machine.models.base_score import BaseScore
from topicnet.cooking_machine.dataset import (
    Dataset,
    VW_TEXT_COL, RAW_TEXT_COL,
    DEFAULT_ARTM_MODALITY, MODALITY_START_SYMBOL
)


class TextType(Enum):
    VW_TEXT = VW_TEXT_COL
    RAW_TEXT = RAW_TEXT_COL


class WordTopicRelatednessType(IntEnum):
    """
    Word-topic relatedness estimate

    Attributes
    ----------
        PWT :
            p(w | t)
        PTW :
            p(t | w)
    """
    PWT = auto()
    PTW = auto()


class SpecificityEstimationMethod(IntEnum):
    """
    Way to estimate how particular word is specific for particular topic.
    Unlike probability, eg. p(w | t), specificity_estimation takes into account
    values for all topics, eg. p(w | t_1), p(w | t_2), ..., p(w | t_n):
    the higher the value p(w | t) comparing other p(w | t_i),
    the higher the specificity_estimation of word "w" for the topic "t"

    Attributes
    ----------
        NONE :
            Don't try to estimate specificity_estimation, return the probability as is
        MAXIMUM :
            From probability, corresponding to word and topic,
            extract *maximum* among probabilities for the word and other topics
        AVERAGE :
            From probability, corresponding to word and topic,
            extract *average* among probabilities for the word and other topics
    """
    NONE = auto()
    MAXIMUM = auto()
    AVERAGE = auto()


class BaseCoherenceScore(BaseScore):
    """"""
    def __init__(
            self,
            dataset: Dataset,
            documents: List[str] = None,
            text_type: TextType = TextType.VW_TEXT,
            word_topic_relatedness: WordTopicRelatednessType = WordTopicRelatednessType.PWT,
            specificity_estimation: SpecificityEstimationMethod = SpecificityEstimationMethod.NONE,
    ):
        """
        Parameters
        ----------
        dataset : Dataset
            Dataset with document collection
            (any model passed to `call()` is supposed to be trained on it)
        documents : list of str
            Which documents from the dataset are to be used for computing coherence
        text_type : TextType
            What text to use when computing coherence: raw text or VW text
            Preferable to use VW (as it is usually preprocessed, stop-words removed etc.),
            and with words in *natural order*.
            Score needs "real" text to compute coherence
        word_topic_relatedness : WordTopicRelatednessType
            How to estimate word relevance to topic: using p(w | t) or p(t | w)
        specificity_estimation : SpecificityEstimationMethod
            How to estimate specificity of word to topic
        """
        # TODO: word_topic_relatedness seems to be connected with TopTokensViewer stuff
        super().__init__()

        if not isinstance(dataset, Dataset):
            raise TypeError(
                f'Got "{type(dataset)}" as \"dataset\". Expect it to derive from "Dataset"')

        if not isinstance(text_type, TextType):
            raise TypeError(
                f'Wrong "text_type": \"{text_type}\". '
                f'Expect to be \"{TextType}\"')

        if not isinstance(word_topic_relatedness, WordTopicRelatednessType):
            raise TypeError(
                f'Wrong "word_topic_relatedness": \"{word_topic_relatedness}\". '
                f'Expect to be \"{WordTopicRelatednessType}\"')

        if not isinstance(specificity_estimation, SpecificityEstimationMethod):
            raise TypeError(
                f'Wrong "specificity_estimation": \"{specificity_estimation}\". '
                f'Expect to be \"{SpecificityEstimationMethod}\"')

        self._dataset = dataset
        self._text_type = text_type
        self._word_topic_relatedness = word_topic_relatedness
        self._specificity_estimation_method = specificity_estimation

        if documents is not None:
            self._documents = documents
        else:
            self._documents = list(self._dataset.get_dataset().index)

    def call(self, model):
        self.topic_coherences = self.compute(model, None)

        coherence_values = list(
            v if v is not None else 0.0  # TODO: state the behavior clearer somehow
            for v in self.topic_coherences.values()
        )

        return np.median(coherence_values)  # TODO: or mean?

    def compute(self, model, topics=None, documents=None):
        if not isinstance(model, BaseModel):
            raise TypeError(
                f'Got "{type(model)}" as "model". '
                f'Expect it to derive from "BaseModel"')

        if topics is None:
            topics = BaseCoherenceScore._get_topics(model)

        if documents is None:
            documents = list(self._documents)

        if not isinstance(topics, list):
            raise TypeError(
                f'Got "{type(topics)}" as "topics". Expect list of topic names')

        if not isinstance(documents, list):
            raise TypeError(
                f'Got "{type(documents)}" as "documents". Expect list of document ids')

        word_topic_relatednesses = self._get_word_topic_relatednesses(model)

        topic_document_coherences = np.zeros((len(topics), len(documents)))
        document_indices_with_topic_coherence = defaultdict(list)

        for document_index, document in enumerate(documents):
            for topic_index, topic in enumerate(topics):
                # TODO: read document text only once for all topics
                topic_coherence = self._compute_coherence(
                    topic, document, word_topic_relatednesses)

                if topic_coherence is not None:
                    topic_document_coherences[topic_index, document_index] = topic_coherence
                    document_indices_with_topic_coherence[topic].append(document_index)

        topic_coherences = [
            topic_document_coherences[topic_index, document_indices_with_topic_coherence[topic]]
            if len(document_indices_with_topic_coherence) > 0 else list()
            for topic_index, topic in enumerate(topics)
        ]

        return dict(zip(
            topics,
            [np.mean(coherence_values) if len(coherence_values) > 0 else None
             for coherence_values in topic_coherences]
        ))

    @staticmethod
    def _get_topics(model):
        return list(model.get_phi().columns)

    def _get_word_topic_relatednesses(self, model):
        phi = model.get_phi()

        word_topic_probs = self._get_word_topic_probs(phi)

        if self._specificity_estimation_method == SpecificityEstimationMethod.NONE:
            pass

        elif self._specificity_estimation_method == SpecificityEstimationMethod.AVERAGE:
            word_topic_probs[:] = (
                word_topic_probs.values -
                    np.sum(word_topic_probs.values, axis=1, keepdims=True) /  # noqa: line alignment
                        max(word_topic_probs.shape[1], 1)
            )

        elif self._specificity_estimation_method == SpecificityEstimationMethod.MAXIMUM:
            new_columns = []

            for t in word_topic_probs.columns:
                new_column = (
                    word_topic_probs[t].values -
                    np.max(
                        word_topic_probs[word_topic_probs.columns.difference([t])].values, axis=1)
                )
                new_columns.append(list(new_column))

            word_topic_probs[:] = np.array(new_columns).T

        return word_topic_probs

    def _get_word_topic_probs(self, phi):
        if self._word_topic_relatedness == WordTopicRelatednessType.PWT:
            return phi

        elif self._word_topic_relatedness == WordTopicRelatednessType.PTW:
            # Treat all topics as equally probable
            eps = np.finfo(float).tiny

            pwt = phi
            pwt_values = pwt.values

            return pd.DataFrame(
                index=pwt.index,
                columns=pwt.columns,
                data=pwt_values / (pwt_values.sum(axis=1).reshape(-1, 1) + eps)
            )

        assert False

    def _compute_coherence(self, topic, document, word_topic_relatednesses):
        raise NotImplementedError()

    def _get_words(self, document):
        def get_biggest_modality_or_default():
            modalities = list(self._dataset.get_possible_modalities())

            if len(modalities) == 0:
                return DEFAULT_ARTM_MODALITY

            modalities_vocabulary_sizes = list(map(
                lambda m: self._dataset.get_dataset().loc[m].shape[0],
                modalities
            ))

            return modalities[np.argmax(modalities_vocabulary_sizes)]

        if self._text_type == TextType.RAW_TEXT:
            text = self._dataset.get_source_document(document).values[0, 0]  # TODO: this way?
            modality = get_biggest_modality_or_default()

            return list(map(lambda w: (modality, w), text.split()))

        if self._text_type == TextType.VW_TEXT:
            text = self._dataset.get_vw_document(document).values[0, 0]  # TODO: this way?

            words = []
            modality = None

            # TODO: there was similar bunch of code somewhere...
            for word in text.split()[1:]:  # skip document id
                if word.startswith(MODALITY_START_SYMBOL):
                    modality = word[1:]

                    continue

                word = word.split(':')[0]

                if modality is not None:
                    word = (modality, word)  # phi multiIndex
                else:
                    word = (DEFAULT_ARTM_MODALITY, word)

                words.append(word)

            return words

        assert False

    @staticmethod
    def _get_relatedness(word, topic, word_topic_relatednesses):
        if word in word_topic_relatednesses.index:
            return word_topic_relatednesses.loc[word, topic]

        # TODO: throw Warning or log somewhere?
        return np.mean(word_topic_relatednesses.values)
