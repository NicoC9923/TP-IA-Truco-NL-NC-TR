# Frozen population ranking reliability diagnostic

We will measure two independent league fitness rankings for each of three schedule splits at 16 and 32 mirror pairs, and compare each selected top eight against a separate 64-pair fitness measurement of the same frozen population. Schedule random streams and odd training-partition deal seeds are independent of training, validation and panel seeds. The diagnostic uses the checkpoint configuration, population, reigning champion and hall, and does not train or promote an individual. Top-eight reference scores average both selections across splits; ties use stable population order.

The pre-registered decision is to proceed to a 32-pair training run only if mean Spearman correlation increases by at least 0.10 and mean reference fitness of the selected top eight is higher at 32 than 16 pairs. Correlations use average ranks for ties; undefined correlations cannot satisfy the rule. A single population and finite reference measurement limit generalization.

The trainer's league schedule and fitness evaluation are extracted into reusable functions with the same RNG order and results, checked against a golden generation log. No training defaults or fitness rules change.
