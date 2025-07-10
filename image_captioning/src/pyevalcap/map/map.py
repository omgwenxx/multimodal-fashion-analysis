import pandas as pd
import stanza
import os
from tqdm import tqdm
import re

class Map:
    """
    Class for computing mean average precision (mAP)
    """

    def __init__(self):
        pass

    def calculate_pr(self, hypo, refe, attrs):
        """
        Calculate precision and recall for a single caption
        refe: ground truth
        hypo: predictions
        """
        hypo_ = set(hypo["caption"].split(" "))  # set to avoid duplicates, one time appearance is enough
        positive = set(refe["caption"].split(" "))  # only contains attributes, all positive = tp + fn
        true_p = [x for x in hypo_ if x in positive]  # tp
        selected = [x for x in hypo_ if x in attrs]  # tp + fp
        prec = len(true_p) / len(selected) if len(selected) != 0 else 0
        reca = len(true_p) / len(positive) if len(positive) != 0 else 0

        return prec, reca, true_p, selected

    def compute_score(
        self,
        gts: list[dict],
        res: list[dict],
        attrs
    ):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        hypotheses = []
        references = []
        tps = []
        selcs = []
        precisions = []
        recalls = []

        for id in tqdm(imgIds, desc="Processing captions"):
            hypo = res[id]
            ref = gts[id]

            # if ref is empty, skip
            # this happens when item has no attributes, only 1 item 0512988022
            if len(ref) == 0:
                print(f"Skipping {id} due to empty reference")
                continue
            
            prec, rec, true_p, selected = self.calculate_pr(hypo, ref, attrs)
            hypotheses.append(set(hypo["caption"]))
            references.append(ref["caption"])
            tps.append(true_p)
            selcs.append(selected)
            precisions.append(prec)
            recalls.append(rec)

        data = {
            "id": imgIds,
            "hypothesis": hypotheses,
            "reference": references,
            "TPs": tps,
            "Selected": selcs,
            "precision": precisions,
            "recall": recalls,
        }

        return pd.DataFrame(data)

    def method(self):
        return "mAP"
