"""
Wrapper for evaluation on CIDEr, ROUGE_L, METEOR, SPICE and Bleu_N
using coco-caption repo https://github.com/tylin/coco-caption
"""

from pyevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pyevalcap.bleu.bleu import Bleu
from pyevalcap.meteor.meteor import Meteor
from pyevalcap.rouge.rouge import Rouge
from pyevalcap.cider.cider import Cider
from pyevalcap.spice.spice import Spice


class Scorer(object):
    """
    Scorer class that computes CIDEr, ROUGE_L, METEOR, SPICE and Bleu_N.
    """

    def __init__(self):
        self.eval = {}
        self.imgToEval = {}
        print("init COCO-EVAL scorer")

    def evaluate(self, GT, RES, IDs):
        """
        This method takes in ground truth (GT) and generated results (RES) along with a list of image IDs (IDs)
        to perform evaluation using various metrics such as BLEU, METEOR, ROUGE_L, CIDEr, and SPICE.

        The evaluation process involves tokenization of captions, setting up scorers, and computing scores for each metric.

        :param GT: Ground truth captions, where keys are image IDs and values are lists of dictionaries
        with the key "caption" and the value being the caption.
        :type GT: dict
        :param RES: Results captions, where keys are image IDs and values are lists of dictionaries
        with the key "caption" and the value being the caption.
        :type RES: dict
        :param IDs: List of image IDs for which evaluation will be performed.
        :type IDs: list
        :return: Dictionary containing evaluation scores for different metrics.
        :rtype: dict

        Example::
            evaluator = Scorer()
            evaluation_results = evaluator.evaluate(ground_truth_captions, generated_captions, image_ids_list)
        """
        gts = {}
        res = {}
        for ID in IDs:
            # print(GT[ID])
            gts[ID] = GT[ID]
            res[ID] = RES[ID]

        print("tokenization...")
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print("setting up scorers...")
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print(f"computing {scorer.method()} score...")
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, IDs, m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, IDs, method)
                print("%s: %0.3f" % (method, score))

        return self.eval

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if imgId not in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score


def score(ref, sample):
    """
    Compute evaluation scores for captions using various metrics.

    Parameters:
        ref (dict): A dictionary containing reference translations.
        sample (dict): A dictionary containing sample translations to be evaluated.

    Returns:
        dict: A dictionary containing final evaluation scores for different metrics.

    Example:
        >>> ref = {'reference': ['This is a reference sentence.']}
        >>> sample = {'translation': ['This is a generated sentence.']}
        >>> result = score(ref, sample)
        >>> print(result)
        {'Bleu_1': 0.5, 'Bleu_2': 0.3, 'Bleu_3': 0.2, 'Bleu_4': 0.1, 'ROUGE_L': 0.4, 'CIDEr': 0.6}

    Note:
        This function uses multiple scoring metrics, including BLEU (1-4), ROUGE_L, and CIDEr. But can be adjusted accordingly.
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    final_scores = {}
    for scorer, method in scorers:
        print("computing %s score with ..." % (scorer.method()))
        score, scores = scorer.compute_score(ref, sample)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores
