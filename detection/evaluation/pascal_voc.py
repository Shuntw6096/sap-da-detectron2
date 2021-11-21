from detectron2.evaluation import PascalVOCDetectionEvaluator
class PascalVOCDetectionEvaluator_(PascalVOCDetectionEvaluator):
    # add AP50 for each class
    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        from detectron2.utils import comm
        from collections import OrderedDict, defaultdict
        import tempfile
        import os
        import numpy as np
        from detectron2.evaluation.pascal_voc_evaluation import voc_eval
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 100, 5):
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                    )
                    aps[thresh].append(ap * 100)

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        # https://github.com/facebookresearch/detectron2/blob/5c923c64c4f4f79a8b1f265be4b6d9d8512b5793/detectron2/evaluation/testing.py#L23
        # '-' in key will not show in logging
        ret["AP50_for_class"] = dict(zip(list(map(lambda x: x.replace('-', '_'), self._class_names)), aps[50]))
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
        return ret
