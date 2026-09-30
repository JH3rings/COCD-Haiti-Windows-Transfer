# Final architecture

Paired ascending/descending observations share a Teacher encoder and FPN. Target and counter features generate the Teacher search policy. Target-only projections provide every searched value, and the decoder consumes only the retrieved target representation. The deployed Student encodes one target orbit and predicts its own policy.

Policy distillation compares bilinearly splatted spatial sampling distributions with Jensen-Shannon divergence. Evidence distillation applies both policies to the same projected Student target memory and compares their retrieved representations by cosine distance. Detached target memory and features restrict the KD gradient to the Student context/policy branch; segmentation trains the complete Student.

The implementation uses three feature levels and four candidates per level, joint candidate weights, and bounded offsets. Geometry masks define evaluation subsets only. Counter-guided selection and aggregation is the mechanism studied; large spatial displacement is not required by the experimental interpretation.
