# Model description

## Counter-guided Target Evidence Search Teacher (CG10)

The Teacher encodes target and counter 5-channel orbit stacks.  Counter
features enter **only** the policy-context path.  Target features are the only
value memory supplied to the deformable search operation.  The policy predicts
sampling offsets and attention weights, retrieves target evidence, and the
decoder receives the retrieved representation `R` only.  Thus this is neither
raw dual-orbit fusion nor a target-feature bypass.  CG10 applies TGD
supervision with `gamma=0.10`.

Implementation: `cocd/windows_main/models_teacher_target_search_v2.py` and
`teacher_target_search_v2.py`.

## Target-only Search Student (S2)

S2 accepts only one target-orbit pre/post stack.  Its policy searches its own
target feature pyramid; the decoder is likewise reachable only via retrieved
evidence `R`.  At deployment it never receives the counter orbit, TGD mask or
Teacher prediction.

Implementation: `cocd/windows_main/models_student_target_search_v2.py`.

## Search distillation

`L_policy` is Jensen-Shannon divergence between Teacher and Student search
distributions. `L_evidence` is cosine distance between the evidence retrieved
from the same stop-gradient Student target memory using Teacher versus Student
policies.  Both are TGD-weighted and fixed-scale normalized.  The final model
uses equal internal weights and a 10% initial KD-to-segmentation contribution.

Implementation: `cocd/windows_main/student_search_kd_v2.py`.
